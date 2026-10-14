#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"
#import "ggml-metal-fusion.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy;  // for async copies
    ggml_metal_event_t ev_sync; // destination completion signal

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_concurrency;
    bool use_graph_optimize;
    bool cb_serial; // encode the command buffers one after another on the calling thread

    int debug_graph;
    int profile_ops; // GGML_METAL_PROFILE_OPS: per-op GPU time, each op encoded in its own command buffer

    // GGML_METAL_PROFILE_CB: command buffer timeline per graph, collected when the next graph starts
    int     profile_cb;
    int64_t pcb_n;
    double  pcb_end_prev; // GPU end time of the last command buffer of the previous graph
    double  pcb_busy;     // GPU time of the graph's command buffers
    double  pcb_gap_in;   // GPU idle between the command buffers of a graph (encode bubble)
    double  pcb_gap_out;  // GPU idle between graphs (host work, uploads)
    double  pcb_enc_main; // host time to encode + commit the main thread's command buffer
    double  pcb_enc_rest; // host time to encode + commit the remaining command buffers

    struct ggml_metal_fusion_info * finfo;

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;
    int cb_end[GGML_METAL_MAX_COMMAND_BUFFERS]; // serial encoding: end node of each of the n_cb command buffers

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // buffers to release after async Metal operations complete
    // if Metal released them, it would do so on a Metal-internal thread without an autorelease pool, which could cause leaks
    NSMutableArray * buf_refs;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;
};

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

    @autoreleasepool {
#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
        // Show all the Metal device instances in the system
        NSArray * devices = MTLCopyAllDevices();
        for (id<MTLDevice> device in devices) {
            GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
        }
        [devices release]; // since it was created by a *Copy* C method
#endif

        // init context
        ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

        id<MTLDevice> device = ggml_metal_device_get_obj(dev);

        GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

        // TODO: would it be better to have one queue for the backend and one queue for the device?
        //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
        //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
        if (queue == nil) {
            GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            free(res);
            return NULL;
        }

        res->dev = dev;
        res->lib = ggml_metal_device_get_library(dev);
        if (res->lib == NULL) {
            GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
            GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

            res->lib = ggml_metal_library_init(dev);
            if (res->lib == NULL) {
                GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

                free(res);

                return NULL;
            }
        }

        res->ev_cpy  = ggml_metal_device_event_init(dev);
        res->ev_sync = ggml_metal_device_event_init(dev);

        const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

        snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

        res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

        // MTLDispatchTypeConcurrent + memoryBarrierWithScope gives wrong results on AMD GPUs
        // (Radeon Pro 5300M, Radeon Pro 5700 XT, see issue 19563), so only use it on Apple GPU families.
        // GGML_METAL_CONCURRENCY_FORCE is an experimental escape hatch to re-enable concurrency on
        // non-Apple GPUs for benchmarking only (expected to produce garbage output on AMD; not for use).
        res->use_concurrency = (getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil) &&
            (props_dev->gpu_family > 0 || getenv("GGML_METAL_CONCURRENCY_FORCE") != nil);

        // non-Apple GPUs: split the rest of the graph into several command buffers that the calling thread
        // encodes in order, committing each one when it is encoded. the GPU then starts on the first while the
        // others are still being encoded, instead of idling until the whole graph is encoded (parallel encoding
        // threads do not help there). GGML_METAL_CB_SERIAL=0/1 overrides
        res->cb_serial = props_dev->gpu_family == 0;
        if (getenv("GGML_METAL_CB_SERIAL")) {
            res->cb_serial = atoi(getenv("GGML_METAL_CB_SERIAL")) != 0;
        }

        {
            const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
            res->debug_graph = val ? atoi(val) : 0;
        }

        {
            const char * val = getenv("GGML_METAL_PROFILE_OPS");
            res->profile_ops = val ? atoi(val) : 0;
        }

        {
            const char * val = getenv("GGML_METAL_PROFILE_CB");
            res->profile_cb   = val ? atoi(val) : 0;
            res->pcb_n        = 0;
            res->pcb_end_prev = 0.0;
            res->pcb_busy     = 0.0;
            res->pcb_gap_in   = 0.0;
            res->pcb_gap_out  = 0.0;
            res->pcb_enc_main = 0.0;
            res->pcb_enc_rest = 0.0;
        }

        res->use_graph_optimize = true;

        if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
            res->use_graph_optimize = false;
        }

        res->finfo = ggml_metal_device_get_fusion_info(dev);
        if (ggml_metal_fusion_info_stats(res->finfo)) {
            ggml_metal_fusion_info_labels_init(res->finfo);
            res->n_cb = 0;
        }

        GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, ggml_metal_fusion_info_enabled(res->finfo) ? "true" : "false");
        GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
        GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

        res->capture_compute = 0;
        res->capture_started = false;
        res->capture_scope = nil;

        {
            const char * val = getenv("GGML_METAL_CAPTURE_COMPUTE");
            if (val) {
                res->capture_compute = atoi(val);
            }
        }

        res->has_error = false;

        res->gf = nil;
        res->encode_async = nil;
        for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
            res->cmd_bufs[i].obj = nil;
        }

        res->cmd_bufs_ext = [[NSMutableArray alloc] init];
        res->buf_refs     = [[NSMutableArray alloc] init];

        res->cmd_buf_last = nil;

        res->pipelines_ext = ggml_metal_pipelines_init();

        return res;
    }
}

static void ggml_metal_prof_print(void);
static void ggml_metal_pcb_collect(ggml_metal_t ctx);

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    if (ctx->profile_ops) {
        ggml_metal_prof_print();
    }

    if (ctx->profile_cb) {
        if (ctx->cmd_buf_last) {
            [ctx->cmd_buf_last waitUntilCompleted];
        }
        ggml_metal_pcb_collect(ctx);

        const double n = ctx->pcb_n > 0 ? (double) ctx->pcb_n : 1.0;
        GGML_LOG_INFO("%s: command buffer timeline over %lld graphs (ms/graph): GPU busy %.3f, gap inside graph %.3f, "
                      "gap between graphs %.3f; host encode main %.3f, rest %.3f\n", __func__, (long long) ctx->pcb_n,
                      1e3*ctx->pcb_busy/n, 1e3*ctx->pcb_gap_in/n, 1e3*ctx->pcb_gap_out/n, 1e3*ctx->pcb_enc_main/n, 1e3*ctx->pcb_enc_rest/n);
    }

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
        [ctx->buf_refs release];
    }

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);

        const int n_fusions = ggml_metal_fusion_info_n_fusions(ctx->finfo);
        for (int i = 0; i < n_fusions; i++) {
            const uint64_t count = ggml_metal_fusion_info_count(ctx->finfo, i);
            if (count == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_metal_fusion_info_label(ctx->finfo, i), count);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);
    ggml_metal_device_event_free(ctx->dev, ctx->ev_sync);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        [ctx->cmd_buf_last waitUntilCompleted];
        ctx->cmd_buf_last = nil;
    }

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ctx->has_error = true;
                return;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_src];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

        // the pages of the caller's memory, or a staging buffer copied out below if Metal refuses them
        size_t offs_dst = 0;
        id<MTLBuffer> buf_dst = ggml_metal_device_wrap_host(ctx->dev, data, size, &offs_dst);
        const bool staged = buf_dst == nil;
        if (staged) {
            offs_dst = 0;
            buf_dst  = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        }

        GGML_ASSERT(buf_dst);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:offs_dst
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        if (staged) {
            // the copy out of the staging buffer needs the blit (and the work queued before it) to be done
            [cmd_buf waitUntilCompleted];
            memcpy(data, [buf_dst contents], size);
            [buf_dst release];
            return;
        }

        [ctx->buf_refs addObject:buf_dst];
        [buf_dst release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        id<MTLCommandQueue> dst_queue = ggml_metal_device_get_queue(ctx_dst->dev);
        id<MTLCommandBuffer> sync_cmd_buf = [dst_queue commandBuffer];

        ggml_metal_event_encode_signal(ctx_dst->ev_sync, sync_cmd_buf);

        [sync_cmd_buf commit];

        [ctx_dst->cmd_bufs_ext addObject:sync_cmd_buf];
        ctx_dst->cmd_buf_last = sync_cmd_buf;

        [sync_cmd_buf retain];

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_event_encode_wait(ctx_dst->ev_sync, cmd_buf);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

// GGML_METAL_PROFILE_OPS: GPU time per op (and per mul_mat type/shape), summed over all graphs,
// printed and reset when the context is freed. Each op is committed and waited on separately,
// so wall-clock throughput is meaningless in this mode - only the per-op shares are.
#define GGML_METAL_PROF_MAX 512

struct ggml_metal_prof_entry {
    char    key[112];
    double  t;
    int64_t n;
};

static struct ggml_metal_prof_entry g_prof[GGML_METAL_PROF_MAX];
static int     g_prof_n      = 0;
static double  g_prof_total  = 0.0;
static int64_t g_prof_graphs = 0;

static void ggml_metal_prof_add(const char * key, double t) {
    g_prof_total += t;

    for (int i = 0; i < g_prof_n; ++i) {
        if (strcmp(g_prof[i].key, key) == 0) {
            g_prof[i].t += t;
            g_prof[i].n += 1;
            return;
        }
    }

    if (g_prof_n < GGML_METAL_PROF_MAX) {
        snprintf(g_prof[g_prof_n].key, sizeof(g_prof[g_prof_n].key), "%s", key);
        g_prof[g_prof_n].t = t;
        g_prof[g_prof_n].n = 1;
        g_prof_n++;
    }
}

static int ggml_metal_prof_cmp(const void * a, const void * b) {
    const double ta = ((const struct ggml_metal_prof_entry *) a)->t;
    const double tb = ((const struct ggml_metal_prof_entry *) b)->t;
    return ta < tb ? 1 : (ta > tb ? -1 : 0);
}

static void ggml_metal_prof_print(void) {
    if (g_prof_n == 0 || g_prof_total <= 0.0) {
        return;
    }

    qsort(g_prof, g_prof_n, sizeof(g_prof[0]), ggml_metal_prof_cmp);

    GGML_LOG_INFO("%s: GPU time over %lld graphs: %.3f ms total, %.3f ms/graph\n", __func__,
            (long long) g_prof_graphs, g_prof_total*1e3, g_prof_total*1e3/MAX(1, g_prof_graphs));

    for (int i = 0; i < g_prof_n; ++i) {
        GGML_LOG_INFO("%s: %6.2f%% %10.3f ms %8lld x  %s\n", __func__,
                100.0*g_prof[i].t/g_prof_total, g_prof[i].t*1e3, (long long) g_prof[i].n, g_prof[i].key);
    }

    g_prof_n      = 0;
    g_prof_total  = 0.0;
    g_prof_graphs = 0;
}

static enum ggml_status ggml_metal_graph_compute_profile(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        if (ctx->cmd_buf_last) {
            [ctx->cmd_buf_last waitUntilCompleted];
            ctx->cmd_buf_last = nil;
        }

        ggml_metal_device_rsets_keep_alive(ctx->dev);

        // walk the same filtered node list as ggml_metal_op (empty / no-op nodes are skipped),
        // so that each op is encoded exactly once and res (fused node count) indexes this list
        int * idxs = malloc(sizeof(int)*gf->n_nodes);
        int n_idxs = 0;
        for (int i = 0; i < gf->n_nodes; ++i) {
            if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
                idxs[n_idxs++] = i;
            }
        }

        for (int p = 0; p < n_idxs; ) {
            const int i = idxs[p];

            id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

            ggml_metal_op_t ctx_op = ggml_metal_op_init(ctx->dev, cmd_buf, gf, ctx->finfo, i, gf->n_nodes, false, false, ctx->debug_graph);
            const int res = ggml_metal_op_encode(ctx_op, 0);
            ggml_metal_op_free(ctx_op);

            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];

            if ([cmd_buf status] != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: command buffer failed at node %d (%s)\n", __func__, i, ggml_op_desc(gf->nodes[i]));
                free(idxs);
                ctx->has_error = true;
                return GGML_STATUS_FAILED;
            }

            if (res == 0) {
                break;
            }

            const struct ggml_tensor * node = gf->nodes[i];

            char key[112];
            if (node->op == GGML_OP_MUL_MAT) {
                snprintf(key, sizeof(key), "MUL_MAT %s M=%lld N=%lld K=%lld", ggml_type_name(node->src[0]->type),
                        (long long) node->src[0]->ne[1], (long long) node->src[1]->ne[1], (long long) node->src[0]->ne[0]);
            } else if (node->op == GGML_OP_MUL_MAT_ID) {
                snprintf(key, sizeof(key), "MUL_MAT_ID %s M=%lld K=%lld E=%lld tok=%lld", ggml_type_name(node->src[0]->type),
                        (long long) node->src[0]->ne[1], (long long) node->src[0]->ne[0], (long long) node->src[0]->ne[2],
                        (long long) node->src[2]->ne[1]);
            } else {
                snprintf(key, sizeof(key), "%s%s", ggml_op_desc(node), res > 1 ? " (fused)" : "");
            }

            ggml_metal_prof_add(key, [cmd_buf GPUEndTime] - [cmd_buf GPUStartTime]);

            p += res;
        }

        free(idxs);

        g_prof_graphs++;
    }

    return GGML_STATUS_SUCCESS;
}

// GPU timeline of the previous graph's command buffers (main thread's first, then cmd_bufs[0..n_cb))
static void ggml_metal_pcb_collect(ggml_metal_t ctx) {
    id<MTLCommandBuffer> cb_main = ctx->cmd_bufs[ctx->n_cb].obj;
    if (!cb_main || [cb_main status] != MTLCommandBufferStatusCompleted) {
        return;
    }

    double busy  = cb_main.GPUEndTime - cb_main.GPUStartTime;
    double gap   = 0.0;
    double e_last = cb_main.GPUEndTime;

    for (int i = 0; i < ctx->n_cb; ++i) {
        id<MTLCommandBuffer> cb = ctx->cmd_bufs[i].obj;
        if (!cb || [cb status] != MTLCommandBufferStatusCompleted) {
            continue;
        }
        gap   += MAX(0.0, cb.GPUStartTime - e_last);
        busy  += cb.GPUEndTime - cb.GPUStartTime;
        e_last = MAX(e_last, cb.GPUEndTime);
    }

    if (ctx->pcb_end_prev > 0.0) {
        ctx->pcb_gap_out += MAX(0.0, cb_main.GPUStartTime - ctx->pcb_end_prev);
    }

    ctx->pcb_end_prev = e_last;
    ctx->pcb_busy    += busy;
    ctx->pcb_gap_in  += gap;
    ctx->pcb_n       += 1;
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    if (gf->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

    if (ctx->profile_ops) {
        return ggml_metal_graph_compute_profile(ctx, gf);
    }

    if (ctx->profile_cb) {
        ggml_metal_pcb_collect(ctx);
    }

    const int64_t pcb_t0 = ctx->profile_cb ? ggml_time_us() : 0;

    // number of nodes encoded by the main thread (empirically determined)
    // serial encoding: a short first command buffer gets the GPU going sooner
    const int n_main = ctx->cb_serial ? 32 : MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    // (cb_serial: the calling thread encodes and commits the n_cb command buffers in order instead)
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        if (ctx->n_cb == 0) {
            // single-threaded encoding: the whole graph is encoded by one command buffer
            ctx->n_nodes_0      = gf->n_nodes;
            ctx->n_nodes_1      = 0;
            ctx->n_nodes_per_cb = 0;
        } else {
            ctx->n_nodes_0      = MIN(n_main, gf->n_nodes);
            ctx->n_nodes_1      = gf->n_nodes - ctx->n_nodes_0;

            ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

            if (ctx->cb_serial) {
                // growing command buffers: each one is encoded while the GPU runs the previous ones
                // (encoding a node is ~5x faster than running it in a decode graph)
                int end = ctx->n_nodes_0;
                int len = ctx->n_nodes_0;
                for (int i = 0; i < ctx->n_cb; ++i) {
                    len *= 3;
                    end  = i == ctx->n_cb - 1 ? gf->n_nodes : MIN(end + len, gf->n_nodes);
                    ctx->cb_end[i] = end;
                }
            }
        }

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s' (did you set METAL_CAPTURE_ENABLED=1 ?)\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        const int64_t pcb_t1 = ctx->profile_cb ? ggml_time_us() : 0;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL || ctx->cb_serial) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        if (ctx->cb_serial) {
            // in order on this thread: each command buffer is committed as soon as it is encoded
            for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
                ctx->encode_async(cb_idx);
            }
        } else {
            dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);
        }

        if (ctx->profile_cb) {
            ctx->pcb_enc_main += 1e-6*(pcb_t1 - pcb_t0);
            ctx->pcb_enc_rest += 1e-6*(ggml_time_us() - pcb_t1);
        }

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            ctx->capture_started = false;
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    // serial encoding: the rest of the graph goes into 4 growing command buffers
    if (ctx->cb_serial && n_cb > 0) {
        n_cb = 4;
    }

    // when fusion stats are collected the graph must be encoded by a single thread so the
    // counters are race-free; override whatever the caller requested
    if (ggml_metal_fusion_info_stats(ctx->finfo)) {
        n_cb = 0;
    }

    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2 && !ctx->cb_serial) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));

            if (ctx->cb_serial) {
                idx_start = cb_idx == 0 ? n_nodes_0 : ctx->cb_end[cb_idx - 1];
                idx_end   = ctx->cb_end[cb_idx];
            }
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            ctx->finfo,
            idx_start,
            idx_end,
            ctx->use_concurrency,
            ctx->capture_compute == 0,
            ctx->debug_graph);

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);

        // the main thread's command buffer (cb_idx == n_cb) is always committed, otherwise the queue stalls
        // behind it; in serial mode nothing is deferred
        if (cb_idx == n_cb_l || cb_idx < 2 || ctx->abort_callback == NULL || ctx->cb_serial) {
            [cmd_buf commit];
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}
