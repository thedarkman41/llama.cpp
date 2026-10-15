#include "ggml-metal-common.h"
#include "ggml-metal-fusion.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
static bool ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
    if (n == 64 || n == 128 || n == 256 || n == 512) {
        return true;
    }

    if (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        return (size_t) n * sizeof(float) <= max_tg_mem;
    }

    return false;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
bool ggml_metal_op_mul_mat_use_fwht(const struct ggml_tensor * op, size_t max_tg_mem) {
    return ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

bool ggml_metal_op_mul_mat_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne11 = op->src[1]->ne[1];

    return !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne21 = op->src[2]->ne[1];

    return has_simdgroup_mm && ne00 >= 64 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
static constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
static constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
static constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

enum ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(enum ggml_type type, int rt) {
    if (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        return GGML_METAL_MMA_KIND_BLK;
    }
    return type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

int ggml_metal_mul_mv_mma_rt(const struct ggml_tensor * op) {
    return op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

static bool ggml_metal_mul_mv_mma_type_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_TQ2_0:
            return true;
        default:
            return false;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(enum ggml_type type, int rt) {
    if (!ggml_metal_mul_mv_mma_type_supported(type)) {
        return 0;
    }
    return ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_size(type) : GGML_METAL_MMA_K_CHUNK;
}

static bool ggml_metal_mul_mat_mma_type_ok(const struct ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    return step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_size(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
static int64_t ggml_metal_mul_mv_mma_rows_min(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return 6;
        case GGML_TYPE_TQ2_0:
            return 5;
        case GGML_TYPE_BF16:
            return 4;
        case GGML_TYPE_F16:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            return 3;
        default:
            return 2;
    }
}

bool ggml_metal_op_mul_mat_use_mma(const struct ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    const bool batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    return ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

bool ggml_metal_op_mul_mat_may_use_mma(const struct ggml_tensor * op) {
    return ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
struct ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

struct ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    int debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(int debug) {
    auto * res = new ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    return res;
}

void ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

static bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    return true;
}

static ggml_mem_range ggml_mem_range_from_tensor(const ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    if (tensor->buffer) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } else {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    return mr;
}

static ggml_mem_range ggml_mem_range_from_tensor_src(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

static ggml_mem_range ggml_mem_range_from_tensor_dst(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

static bool ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

static bool ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    return ggml_mem_ranges_add_dst(mrs, tensor);
}

static bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    for (size_t i = 0; i < mrs->ranges.size(); i++) {
        const auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        if (mr.pb != cmp.pb) {
            continue;
        }

        // intersecting source ranges are allowed
        if (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            continue;
        }

        if (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            if (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            return false;
        }
    }

    return true;
}

static bool ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

static bool ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            if (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                return false;
            }
        }
    }

    return ggml_mem_ranges_check_dst(mrs, tensor);
}

struct node_info {
    ggml_tensor * node;

    std::vector<ggml_tensor *> fused;

    ggml_op op() const {
        return node->op;
    }

    const ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }

    bool is_output(const ggml_tensor * t) const {
        if (t == node) {
            return true;
        }
        for (const auto * f : fused) {
            if (t == f) {
                return true;
            }
        }
        return false;
    }
};

static std::vector<int> ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes) {
    // helper to add node src and dst ranges
    const auto & h_add = [](ggml_mem_ranges_t mrs, const node_info & node) {
        // only external sources matter: sources produced by the fused group are internal
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const ggml_tensor * src = node.node->src[i];
            if (src && !node.is_output(src)) {
                if (!ggml_mem_ranges_add_src(mrs, src)) {
                    return false;
                }
            }
        }

        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                const ggml_tensor * src = fused->src[i];
                if (src && !node.is_output(src)) {
                    if (!ggml_mem_ranges_add_src(mrs, src)) {
                        return false;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        if (!ggml_mem_ranges_add_dst(mrs, node.node)) {
            return false;
        }
        for (const auto * fused : node.fused) {
            if (!ggml_mem_ranges_add_dst(mrs, fused)) {
                return false;
            }
        }

        return true;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    const auto & h_check = [](ggml_mem_ranges_t mrs, const node_info & node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const ggml_tensor * src = node.node->src[i];
            if (src && !node.is_output(src)) {
                if (!ggml_mem_ranges_check_src(mrs, src)) {
                    return false;
                }
            }
        }

        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                const ggml_tensor * src = fused->src[i];
                if (src && !node.is_output(src)) {
                    if (!ggml_mem_ranges_check_src(mrs, src)) {
                        return false;
                    }
                }
            }
        }

        if (!ggml_mem_ranges_check_dst(mrs, node.node)) {
            return false;
        }
        for (const auto * fused : node.fused) {
            if (!ggml_mem_ranges_check_dst(mrs, fused)) {
                return false;
            }
        }

        return true;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    const auto & h_safe = [](ggml_op op) {
        switch (op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
            case GGML_OP_ROPE:
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM:
            case GGML_OP_SUM_ROWS:
            case GGML_OP_SSM_CONV:
            case GGML_OP_SSM_SCAN:
            case GGML_OP_CLAMP:
            case GGML_OP_TRI:
            case GGML_OP_DIAG:
            case GGML_OP_MUL:
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
            case GGML_OP_GLU:
            case GGML_OP_SCALE:
            case GGML_OP_UNARY:
            case GGML_OP_GET_ROWS:
            case GGML_OP_SET_ROWS:
            case GGML_OP_SET:
            case GGML_OP_CPY:
            case GGML_OP_CONT:
            case GGML_OP_REPEAT:
                return true;
            default:
                return ggml_op_is_empty(op);
        }
    };

    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        if (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr int N_FORWARD = 64;

            for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                if (used[i1]) {
                    continue;
                }

                const auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                if (!h_safe(node1.op())) {
                    break;
                }

                const bool is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                if ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = true;
                } else {
                    // expand the set of nodes that haven't been processed yet
                    h_add(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_add(mrs0, node0);
            res.push_back(i0);
        }
    }

    ggml_mem_ranges_free(mrs0);
    ggml_mem_ranges_free(mrs1);

    return res;
}

static const ggml_tensor * ggml_metal_view_root(const ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

// ops that write only their own tensor (the nodes that a decode-GDN gather may be moved past)
static bool ggml_metal_dn_sink_safe(ggml_op op) {
    switch (op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_SSM_CONV:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
        case GGML_OP_GLU:
        case GGML_OP_SCALE:
        case GGML_OP_UNARY:
        case GGML_OP_GET_ROWS:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_REPEAT:
            return true;
        default:
            return ggml_op_is_empty(op);
    }
}

// move the one-row GET_ROWS gather gr and its views up to `use` (the view of gr read by the node at raw index ia) to
// right before that node; zero-size readers of gr (the empty extra-states copy of build_rs) move after raw index
// i_last >= ia. the other nodes in between keep their order and must not write the rows or the ids that gr reads
static bool ggml_metal_dn_sink_gather(ggml_cgraph * gf, int ia, const ggml_tensor * gr, const ggml_tensor * use, int i_last) {
    if (gr->op != GGML_OP_GET_ROWS || ggml_nelements(gr->src[1]) != 1) {
        return false;
    }

    std::vector<const ggml_tensor *> chain; // the views from use back to gr (exclusive)
    for (const ggml_tensor * t = use; t != gr; t = t->src[0]) {
        if (t == nullptr || !ggml_op_is_empty(t->op) || chain.size() >= 2) { // VIEW + RESHAPE, as in ops_gdn_gather
            return false;
        }
        chain.push_back(t);
    }

    int ir = -1;
    for (int j = ia - 1; j >= 0 && j >= ia - 1024; --j) {
        if (gf->nodes[j] == gr) {
            ir = j;
            break;
        }
    }
    if (ir < 0) {
        return false;
    }

    const ggml_tensor * root0 = ggml_metal_view_root(gr->src[0]);
    const ggml_tensor * root1 = ggml_metal_view_root(gr->src[1]);

    std::vector<ggml_tensor *> keep;  // stay in order, before the gather
    std::vector<ggml_tensor *> moved; // the chain views (in graph order), after the gather
    std::vector<ggml_tensor *> tail;  // zero-size readers of the gather, after i_last
    std::vector<const ggml_tensor *> dep = { gr };

    for (int j = ir + 1; j < ia; ++j) {
        ggml_tensor * node = gf->nodes[j];

        bool reads = false;
        for (int s = 0; s < GGML_MAX_SRC && !reads; ++s) {
            reads = node->src[s] && std::find(dep.begin(), dep.end(), node->src[s]) != dep.end();
        }

        if (reads) {
            dep.push_back(node);
            if (std::find(chain.begin(), chain.end(), node) != chain.end()) {
                moved.push_back(node);
            } else if (ggml_op_is_empty(node->op) || ggml_nelements(node) == 0) {
                tail.push_back(node);
            } else {
                return false; // the gathered row has another reader
            }
            continue;
        }

        // the gather moves past this node: it must not write the cache rows or the row ids that the gather reads
        if (!ggml_op_is_empty(node->op)) {
            const ggml_tensor * root = ggml_metal_view_root(node);
            if (!ggml_metal_dn_sink_safe(node->op) || root == root0 || root == root1) {
                return false;
            }
        }

        keep.push_back(node);
    }

    if (moved.size() != chain.size()) {
        return false;
    }

    // the nodes that the tail moves past must not read it
    for (int j = ia; j <= i_last; ++j) {
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (gf->nodes[j]->src[s] && std::find(tail.begin(), tail.end(), gf->nodes[j]->src[s]) != tail.end()) {
                return false;
            }
        }
    }

    // [ir, i_last] -> keep, gather, chain views, [ia, i_last], tail
    std::vector<ggml_tensor *> order;
    order.reserve(i_last - ir + 1);
    order.insert(order.end(), keep.begin(), keep.end());
    order.push_back(const_cast<ggml_tensor *>(gr));
    order.insert(order.end(), moved.begin(), moved.end());
    for (int j = ia; j <= i_last; ++j) {
        order.push_back(gf->nodes[j]);
    }
    order.insert(order.end(), tail.begin(), tail.end());

    GGML_ASSERT((int) order.size() == i_last - ir + 1);

    for (int j = ir; j <= i_last; ++j) {
        gf->nodes[j] = order[j - ir];
    }

    return true;
}

// decode GDN (one token of one sequence): the recurrent state is gathered in build_rs, long before the GDN. move the
// gather next to the GDN, so that GET_ROWS, GATED_DELTA_NET and the trailing cache CPY form the GDN_GATHER fusion
static void ggml_metal_dn_sink_state(ggml_cgraph * gf, int ig) {
    const ggml_tensor * gdn = gf->nodes[ig];
    const ggml_tensor * st  = gdn->src[5];

    if (ggml_get_op_params_i32(gdn, 0) != 1 || gdn->src[2]->ne[2] != 1 || gdn->src[2]->ne[3] != 1 || st->view_src == nullptr) {
        return;
    }

    // the trailing cache cpy of the new state (only views in between)
    int ic = ig + 1;
    while (ic < gf->n_nodes && ggml_op_is_empty(gf->nodes[ic]->op)) {
        ic++;
    }
    if (ic >= gf->n_nodes || gf->nodes[ic]->op != GGML_OP_CPY || gf->nodes[ic]->src[0]->view_src != gdn || ic - ig != 3) {
        return;
    }

    ggml_metal_dn_sink_gather(gf, ig, st->view_src, st, ic);
}

void ggml_graph_canonicalize_dn_decode(ggml_cgraph * gf) {
    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->nodes[i]->op == GGML_OP_GATED_DELTA_NET) {
            ggml_metal_dn_sink_state(gf, i);
        }
    }
}

void ggml_graph_optimize(ggml_cgraph * gf) {
    const int n = gf->n_nodes;

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    for (int i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        const int f = ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        for (int k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            node.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(node));
    }

#if 1
    // reorder to improve concurrency
    const auto order = ggml_metal_graph_optimize_reorder(nodes);
#else
    std::vector<int> order(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
