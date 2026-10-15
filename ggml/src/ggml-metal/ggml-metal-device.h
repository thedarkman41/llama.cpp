#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_buffer_id {
    void * metal; // id<MTLBuffer>
    size_t offs;
};

typedef struct ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

typedef struct ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(void);
void ggml_metal_cv_free(ggml_metal_cv_t cv);

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx);
void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx);
void ggml_metal_cv_set_bool (ggml_metal_cv_t cv, bool    value, int32_t idx);

//
// MTLComputePipelineState wrapper
//

typedef struct ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(void);
void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
typedef struct ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(void);
void ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

void                  ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, const char * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, const char * name);

struct ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    int nsg;

    int nr0;
    int nr1;

    size_t smem;

    bool c4;
    bool cnt;
};

int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

typedef void * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

typedef struct ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
void ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name);
void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder);

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, struct ggml_metal_pipeline_with_params pipeline);

void ggml_metal_encoder_set_bytes (ggml_metal_encoder_t encoder, void * data, size_t size, int idx);
void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx);

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx);

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2);

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder);

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

typedef struct ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, const char * source, bool verbose);

void ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, const char * name);
struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base              (ggml_metal_library_t lib, enum ggml_op op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, enum ggml_type tsrc, enum ggml_type tdst, bool i32);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, const struct ggml_tensor * op, enum ggml_op_pool op_pool);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, const struct ggml_tensor * op, enum ggml_op_pool op_pool);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows          (ggml_metal_library_t lib, enum ggml_type tsrc);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows          (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, enum ggml_type tsrc);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, enum ggml_type tsrc);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat_tr         (ggml_metal_library_t lib, enum ggml_type tsrc);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_back         (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum               (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows          (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add        (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, const struct ggml_tensor * op, int32_t nc, bool use_silu, bool rows);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, const struct ggml_tensor * op, int ssm_conv_bs, int32_t nc, bool use_silu);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_tiled    (ggml_metal_library_t lib, const struct ggml_tensor * op, int32_t nc, bool use_silu);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan          (ggml_metal_library_t lib, const struct ggml_tensor * op, bool tail);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan_ssd_mma  (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, const struct ggml_tensor * op, bool gather);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net_rb(ggml_metal_library_t lib, const struct ggml_tensor * op, int lpr, int rpl);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, const struct ggml_tensor * op, int nsg, int nxpsg, int r1ptg);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, const struct ggml_tensor * op, int nsg, int nt, int rt, bool add);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, const struct ggml_tensor * op, bool add);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_tiled      (ggml_metal_library_t lib, const struct ggml_tensor * op, bool src1_f16);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_wc         (ggml_metal_library_t lib, const struct ggml_tensor * op, bool src1_f16, bool early, int ks, bool dq2);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_wc_ksum    (ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mm_cvt_f32_f16    (ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, int ne02, int ne20);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_tiled   (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map_wl  (ggml_metal_library_t lib, int ne02, int ne20, bool v2);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_wl      (ggml_metal_library_t lib, const struct ggml_tensor * op, int bk, bool dq2, bool b16);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge     (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, int n, enum ggml_type tsrc);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k             (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_radix       (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_merge       (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t top_k, bool with_norm);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe_nd       (ggml_metal_library_t lib, int32_t n_expert, int32_t top_k, bool with_norm, bool minmax);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_used);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, const struct ggml_tensor * op, int32_t n_fuse);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, enum ggml_op op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm        (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, const struct ggml_tensor * op, int32_t n_fuse);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, enum ggml_type type);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, const struct ggml_tensor * op, bool tiled);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw    (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd      (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_count_equal       (ggml_metal_library_t lib, const struct ggml_tensor * op);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool has_mask,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        bool    has_sparse,
        int32_t nqpsg,
        int32_t ne,
        int32_t nsg,
        int32_t nwg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

typedef void * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
typedef struct ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
void ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

enum ggml_metal_device_id {
    GGML_METAL_DEVICE_GENERIC = 0,

    GGML_METAL_DEVICE_M1,
    GGML_METAL_DEVICE_M1_PRO,
    GGML_METAL_DEVICE_M1_MAX,
    GGML_METAL_DEVICE_M1_ULTRA,
    GGML_METAL_DEVICE_M2,
    GGML_METAL_DEVICE_M2_PRO,
    GGML_METAL_DEVICE_M2_MAX,
    GGML_METAL_DEVICE_M2_ULTRA,
    GGML_METAL_DEVICE_M3,
    GGML_METAL_DEVICE_M3_PRO,
    GGML_METAL_DEVICE_M3_MAX,
    GGML_METAL_DEVICE_M3_ULTRA,
    GGML_METAL_DEVICE_M4,
    GGML_METAL_DEVICE_M4_PRO,
    GGML_METAL_DEVICE_M4_MAX,
    GGML_METAL_DEVICE_M5,
    GGML_METAL_DEVICE_M5_PRO,
    GGML_METAL_DEVICE_M5_MAX,
    GGML_METAL_DEVICE_M5_ULTRA,
    GGML_METAL_DEVICE_A18_PRO,
};

const char * ggml_metal_device_id_token(enum ggml_metal_device_id id);

struct ggml_metal_device_props {
    int device;
    int device_phys;
    int device_virt;
    char name[128];
    char desc[128];

    size_t max_buffer_size;
    size_t max_working_set_size;
    size_t max_theadgroup_memory_size;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_memory;
    bool has_bfloat;
    bool has_tensor;
    bool use_residency_sets;
    bool use_shared_buffers;

    enum ggml_metal_device_id device_id;
    int gpu_family;

    int op_offload_min_batch_size;

    int mm_wc; // GGML_METAL_MM_WC: 1 = kernel_mul_mm_wc (default), 0 = kernel_mul_mm_tiled, 2 = kernel_mul_mm_wc with late staging
    int mm_dq2; // GGML_METAL_MM_DQ2: bitmask of the kernel_mul_mm_wc dq2 A loaders: bit 0 q3_K, 1 q4_K, 2 q6_K, 3 iq4_xs, 4 iq3_s, 5 iq3_xxs, 6 iq2_s (default 0x7f); 0 = dequantize_* loaders

    int mv_v2; // GGML_METAL_MV_V2: bitmask of v2 mat-vec bodies on GPUs without simdgroup-mm: bit 0 iq3_s, 1 iq3_xxs, 2 iq2_s, 3 iq4_xs, 4 q5_K nr0=2, 5 q3_K, 6 q4_K (5, 6 dense only); 0 = old kernels

    int moe_wl;     // GGML_METAL_MOE_WL: 2 = kernel_mul_mm_id_wl with BK = 64 (default), 1 = BK = 32, 0 = mul_mv_id
    int moe_wl_min; // GGML_METAL_MOE_WL_MIN: least average routed tokens per expert for kernel_mul_mm_id_wl (default 1), 0 = any batch (testing)
    int moe_map;    // GGML_METAL_MOE_MAP: 1 = kernel_mul_mm_id_map_wl2 (default), 0 = kernel_mul_mm_id_map_wl
    int moe_dq2;    // GGML_METAL_MOE_DQ2: bitmask of the dq2 A loaders in kernel_mul_mm_id_wl (BK = 64), bits as mm_dq2 (default 0x7f); 0 = dequantize_* loaders
    int moe_b16;    // GGML_METAL_MOE_B16: 1 = the dq2 kernels read a contiguous f16 copy of src1 (default), 0 = src1
    int moe_tgg;    // GGML_METAL_MOE_TGG: 1 = dq2 decoder lookup tables (iq2_s, iq3_s, iq3_xxs, iq4_xs) in threadgroup memory (default), 0 = constant memory

    int topk_moe;   // GGML_METAL_TOPK_MOE: 0 = kernel_topk_moe_f32, 1 = kernel_topk_moe_f32_nd, 2 = _nd with simd_max/simd_min select (default without simdgroup-mm)

    int gdn_rb;     // GGML_METAL_GDN_RB: 1 = register-blocked GATED_DELTA_NET (default on GPUs without simdgroup-mm), 0 = off
    int gdn_rb_min; // GGML_METAL_GDN_RB_MIN: least tokens per sequence for the rb kernel (default 2: decode keeps the old kernel)
    int gdn_rb_cfg; // GGML_METAL_GDN_RB_CFG: 0 = auto, else LPR*10 + RPL in {81, 82, 42}

    int dn_decode;  // GGML_METAL_DN_DECODE: 1 = the decode GDN reads its state in place from the recurrent cache (GDN_GATHER, serial encoder only; default on GPUs without simdgroup-mm), 0 = off
    int dn_conv;    // GGML_METAL_DN_CONV: bitmask of the decode conv-chain paths: bit 0 CONCAT with a 1-wide transposed source on the generic kernel, bit 1 CPY with 32-bit indexing, bit 2 one-token SSM_CONV with 256-thread threadgroups (default 0x7 on GPUs without simdgroup-mm); 0 = old paths
};

typedef struct ggml_metal_event * ggml_metal_event_t;

void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
void ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(int device, int n_devices);
void ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(int device, int n_devices);

void * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
void * ggml_metal_device_get_queue(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

void ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset);
void ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
void ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

// staging ring for small host -> device uploads (GGML_METAL_UPLOAD_RING, see ggml-metal-device.m)
// upload: copies data into the ring and blits it to dst (id<MTLBuffer>) at dst_offs. with cmd_buf, the blit is
// committed in its own command buffer, returned retained; without it, mode 2 defers the blit to the next flush.
// returns false if the ring is off (the caller uses its old path)
bool ggml_metal_device_upload(ggml_metal_device_t dev, void * dst, size_t dst_offs, const void * data, size_t size, void ** cmd_buf);
// call before any submission to the device queue. with cmd_buf (id<MTLCommandBuffer>), it enqueues cmd_buf and encodes
// the deferred blits at its head; without it, they get a command buffer of their own
void ggml_metal_device_upload_flush(ggml_metal_device_t dev, void * cmd_buf);
// releases what the ring holds for finished copies
void ggml_metal_device_upload_reclaim(ggml_metal_device_t dev);

void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op);

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

struct ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
struct ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

typedef struct ggml_metal_buffer * ggml_metal_buffer_t;

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared);
ggml_metal_buffer_t ggml_metal_buffer_map (ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size);

void   ggml_metal_buffer_free     (ggml_metal_buffer_t buf);
void * ggml_metal_buffer_get_base (ggml_metal_buffer_t buf);
bool   ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf);

void   ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
void   ggml_metal_buffer_set_tensor   (ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void   ggml_metal_buffer_get_tensor   (ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);

// a shared Metal buffer over the host pages that hold [ptr, ptr + size), for blits to or from host memory:
// newBufferWithBytesNoCopy needs a page-aligned address and length (for other addresses it can return nil,
// seen on a Radeon Pro 5700 XT). *offs is where ptr lands in the buffer. retained; nil if Metal refuses
void * ggml_metal_device_wrap_host(ggml_metal_device_t dev, const void * ptr, size_t size, size_t * offs); // id<MTLBuffer>

bool   ggml_metal_buffer_cpy_tensor   (ggml_metal_buffer_t buf, const struct ggml_tensor * src, struct ggml_tensor * dst);
void   ggml_metal_buffer_clear        (ggml_metal_buffer_t buf, uint8_t value);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t);

#ifdef __cplusplus
}
#endif
