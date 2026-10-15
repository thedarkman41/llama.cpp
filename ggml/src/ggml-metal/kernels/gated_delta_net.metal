#include "common.h"

constant short FC_gated_delta_net_ne20 [[function_constant(FC_GATED_DELTA_NET + 0)]];
constant short FC_gated_delta_net_ne30 [[function_constant(FC_GATED_DELTA_NET + 1)]];
constant short FC_gated_delta_net_K    [[function_constant(FC_GATED_DELTA_NET + 2)]];
constant bool  FC_gated_delta_net_gather [[function_constant(FC_GATED_DELTA_NET + 3)]];

#if 1
template<short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device       char * dst_fuse,
        device const char * ids,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30
#define K   FC_gated_delta_net_K

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B (n_seqs)
    const uint i21 = tgpig.y; // H (head)
    const uint i20 = tgpig.x*NSG + ty; // row within S_v

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    // input state layout [S_v, S_v, H, n_seqs] (s0 only): per-seq stride is H*D.
    // state is stored transposed: M[i20][is] = S[is][i20], so row i20 is contiguous
    const uint state_in_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;
    device const float * s_ptr = (device const float *) (s) + state_in_base;

    if (FC_gated_delta_net_gather) {
        // GDN_GATHER: s is the recurrent cache and the state of seq i23 is its row ids[i23]. each thread reads
        // the same elements that it writes back below, so the in-place update of that row is race-free
        s_ptr = (device const float *) (s) + (uint64_t) ((device const int32_t *) ids)[i23]*args.ns_row + (i21*S_v*S_v + i20*S_v);
    }

    float ls[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        ls[j] = s_ptr[is];
    }

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K, only slots 0..n_tokens-1 are written; older slots are caller-owned.

    // output state base offset: after attention scores
    const uint attn_size = args.ne22 * args.ne21 * S_v * args.ne23;
    // output state per-slot size: S_v * S_v * H * n_seqs
    const uint state_size_per_snap = S_v * S_v * args.ne21 * args.ne23;
    // per-(seq,head) offset within a slot
    const uint state_out_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;

    // when fused with the cache cpy, write the snapshots straight into the cache buffer using
    // the slot stride; otherwise append them after the attn scores (nb_out == 0)
    const bool fused = args.nb_out > 0;
    const device float * state_out = fused ? (device float *)dst_fuse : (device float *)dst + attn_size;
    const uint slot_stride = fused ? (uint)args.nb_out : state_size_per_snap;

    for (short t = 0; t < args.ne22; t++) {
        float s_k = 0.0f;

        if (G == 1) {
            const float g_exp = exp(g_ptr[0]);

            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= g_exp;

                s_k += ls[j]*k_ptr[is];
            }
        } else {
            // KDA
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= exp(g_ptr[is]);

                s_k += ls[j]*k_ptr[is];
            }
        }

        s_k = simd_sum(s_k);

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        float y = 0.0f;

        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            ls[j] += k_ptr[is]*d;

            y += ls[j]*q_ptr[is];
        }

        y = simd_sum(y);

        if (tx == 0) {
            dst_attn[t*args.ne21*S_v] = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        if (K > 1) {
            const int target_slot = (int)args.ne22 - 1 - (int)t;
            if (target_slot >= 0 && target_slot < (int)K) {
                device float * dst_state = (device float *)state_out + (uint)target_slot * slot_stride + state_out_base;
                FOR_UNROLL (short j = 0; j < NSG; j++) {
                    const short is = tx*NSG + j;
                    dst_state[is] = ls[j];
                }
            }
        }
    }

    if (K == 1) {
        device float * dst_state = (device float *)state_out + state_out_base;
        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            dst_state[is] = ls[j];
        }
    }

#undef S_v
#undef G
#undef K
}

typedef decltype(kernel_gated_delta_net_impl<4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<4>;

#else
// a simplified version of the above
// no performance improvement, so keep the above version for now

template<typename T, short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device       char * dst_fuse,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B
    const uint i21 = tgpig.y; // H
    const uint i20 = tgpig.x*NSG + ty;

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    device const float * s_ptr = (device const float *) (s) + (i23*args.ne21 + i21)*S_v*S_v + i20;

    float lsf[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        lsf[j] = s_ptr[is*S_v];
    }

    thread T * ls = (thread T *) (lsf);

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr  = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr  = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    for (short t = 0; t < args.ne22; t++) {
        device const T * qt_ptr = (device const T *) (q_ptr);
        device const T * kt_ptr = (device const T *) (k_ptr);
        device const T * gt_ptr = (device const T *) (g_ptr);

        if (G == 1) {
            *ls *= exp(g_ptr[0]);
        } else {
            // KDA
            *ls *= exp(gt_ptr[tx]);
        }

        const float s_k = simd_sum(dot(*ls, kt_ptr[tx]));

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        *ls += kt_ptr[tx]*d;

        const float y = simd_sum(dot(*ls, qt_ptr[tx]));

        if (tx == 0) {
            *dst_attn = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        dst_attn += args.ne21*S_v;
    }

    // when fused with the cache cpy, write the snapshots straight into the cache buffer using
    // the slot stride; otherwise append them after the attn scores (nb_out == 0)
    const bool fused = args.nb_out > 0;
    const device float * state_out = fused ? (device float *)dst_fuse : (device float *)dst + args.ne23*args.ne22*args.ne21*S_v;
    const uint slot_stride = fused ? (uint)args.nb_out : S_v*S_v;

    device float * dst_state  = (device float *)state_out + (i23*args.ne21 + i21)*slot_stride + i20;
    device T     * dstt_state = (device T     *) (dst_state);

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        dst_state[is*S_v] = lsf[j];
    }

#undef S_v
#undef G
}

typedef decltype(kernel_gated_delta_net_impl<float4, 4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float,  1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float2, 2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float4, 4>;
#endif

// -------------------------------------------------------------------------------------------------
// kernel_gated_delta_net_rb: register-blocked GATED_DELTA_NET for GPUs without simdgroup-mm.
// The host selects it in ggml_metal_op_gdn_rb_select; Apple GPUs keep kernel_gated_delta_net_impl.
//
// One simdgroup per threadgroup. Lane l = rg*LPR + sub. The LPR lanes of row group rg share RPL
// consecutive (transposed) state rows. Lane sub owns float4 column chunks c*LPR + sub (c < NC).
// s_k and y are reduced over the LPR lanes only, with one xor butterfly per token, and the y of
// token t-1 is folded into the s_k butterfly of token t. q/k (and exp(g) for KDA) of token t+1 are
// loaded with coalesced float4 loads during token t, then staged in double-buffered threadgroup
// memory: one barrier per token.
// NOTE: int (not short) for every index/address local; short locals miscompile on AMD here.
template<int SV, int LPR, int RPL>
kernel void kernel_gated_delta_net_rb_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device       char * dst_fuse,
        threadgroup float4 * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {
    constexpr int NK4  = SV/4;                                // float4 per q / k / g / state row
    constexpr int NC   = NK4/LPR;                             // float4 chunks per lane and row
    constexpr int NG   = 32/LPR;                              // row groups per simdgroup
    constexpr int RPW  = NG*RPL;                              // state rows per simdgroup (= per threadgroup)
    constexpr int NPF  = (2*NK4 + 31)/32;                     // prefetched q/k float4 per lane
    constexpr int LLPR = LPR == 8 ? 3 : 2;                    // log2(LPR)

    static_assert(LPR == 4 || LPR == 8, "LPR must be 4 or 8");
    static_assert(NK4 % LPR == 0 && NC >= 1, "SV must be a multiple of 4*LPR");
    static_assert(SV % RPW == 0, "SV must be a multiple of the rows per simdgroup");
    static_assert(RPL <= LPR, "y store uses lanes sub < RPL");
    // RPL = 4 is miscompiled on Radeon Pro 5700 XT: float yp[4] reads element 3 as 0 under the per-lane select
    // of the deferred y store (the state stays exact). Padding yp to RPL + 1 or a switch on sub avoids it
    static_assert(RPL != 4, "see the note above before instantiating RPL = 4");
    static_assert(NK4 <= 32, "KDA staging uses one float4 per lane");

    const int  G    = FC_gated_delta_net_ne30;   // 1 or SV (host-checked)
    const int  KS   = FC_gated_delta_net_K;      // snapshot slots
    const bool kda  = G != 1;
    const int  NBUF = kda ? 3*NK4 : 2*NK4;       // float4 per staging buffer: [k | q | exp(g)]

    const int lane = (int) tpitg.x;
    const int sub  = lane % LPR;
    const int rg   = lane / LPR;

    const int i23 = (int) tgpig.z;                 // seq
    const int i21 = (int) tgpig.y;                 // v head
    const int i20 = (int) tgpig.x*RPW + rg*RPL;    // first of this lane's RPL rows

    const int H = args.ne21;
    const int T = args.ne22;

    const int i01 = i21 % args.ne01;
    const int i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float) SV);

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21) + i20;
    device const float * b_ptr = (device const float *) (b) + (i23*T*H + i21);
    device const float * g_ptr = (device const float *) (g) + (i23*T*H + i21)*G;

    device float * dst_attn = (device float *) (dst) + (uint) (i23*T*H + i21)*SV + i20;

    // same transposed layout and offsets as kernel_gated_delta_net_impl (:40, :67)
    const uint state_base = ((uint) (i23*H + i21)*SV + (uint) i20)*SV;

    float4 ls[RPL][NC];
    {
        device const float4 * s4 = (device const float4 *) ((device const float *) s + state_base);
        FOR_UNROLL (int r = 0; r < RPL; r++) {
            FOR_UNROLL (int c = 0; c < NC; c++) {
                ls[r][c] = s4[r*NK4 + c*LPR + sub];
            }
        }
    }

    const uint attn_size           = (uint) args.ne22*args.ne21*SV*args.ne23;
    const uint state_size_per_snap = (uint) SV*SV*args.ne21*args.ne23;
    const bool fused               = args.nb_out > 0;
    device float * state_out       = fused ? (device float *) dst_fuse : (device float *) dst + attn_size;
    const uint     slot_stride     = fused ? (uint) args.nb_out : state_size_per_snap;

    float4 pf[NPF];        // next-token q/k (cooperative)
    float4 pfg = 0.0f;     // next-token g float4 (KDA)
    float  pv[RPL];        // next-token v of this lane's rows
    float  pgs = 0.0f;     // next-token scalar g (G == 1)
    float  pb  = 0.0f;     // next-token beta
    float  cv[RPL];        // current-token v
    float  cge = 0.0f;     // current-token exp(g) (G == 1)
    float  cb  = 0.0f;     // current-token beta
    float  yp[RPL];        // per-lane partial y of the previous token

    FOR_UNROLL (int r = 0; r < RPL; r++) { pv[r] = 0.0f; cv[r] = 0.0f; yp[r] = 0.0f; }
    FOR_UNROLL (int p = 0; p < NPF; p++) { pf[p] = 0.0f; }

    for (int t = -1; t < T; t++) {
        const int tn = t + 1;

        // 1. global loads of token t+1 (consumed in step 3; latency hidden by step 2)
        if (tn < T) {
            device const float * qn = q_ptr + tn*args.ns02;
            device const float * kn = k_ptr + tn*args.ns12;
            FOR_UNROLL (int p = 0; p < NPF; p++) {
                const int i = lane + 32*p;
                if (i < 2*NK4) {
                    device const float * src = i < NK4 ? kn + 4*i : qn + 4*(i - NK4);
                    pf[p] = *((device const float4 *) src);
                }
            }
            device const float * vn = v_ptr + tn*args.ns22;
            FOR_UNROLL (int r = 0; r < RPL; r++) { pv[r] = vn[r]; }
            pb = b_ptr[tn*H];
            if (kda) {
                if (lane < NK4) {
                    pfg = *((device const float4 *) (g_ptr + tn*H*G) + lane);
                }
            } else {
                pgs = g_ptr[tn*H];
            }
        }

        // 2. token t
        if (t >= 0) {
            const int cur = (t & 1)*NBUF;

            float4 kc[NC];
            float4 qc[NC];
            FOR_UNROLL (int c = 0; c < NC; c++) { kc[c] = shmem[cur +       c*LPR + sub]; }
            FOR_UNROLL (int c = 0; c < NC; c++) { qc[c] = shmem[cur + NK4 + c*LPR + sub]; }

            float sk[RPL];
            if (kda) {
                float4 gd[NC];
                FOR_UNROLL (int c = 0; c < NC; c++) { gd[c] = shmem[cur + 2*NK4 + c*LPR + sub]; }
                FOR_UNROLL (int r = 0; r < RPL; r++) {
                    float4 acc = 0.0f;
                    FOR_UNROLL (int c = 0; c < NC; c++) {
                        ls[r][c] *= gd[c];
                        acc += ls[r][c]*kc[c];
                    }
                    sk[r] = (acc.x + acc.y) + (acc.z + acc.w);
                }
            } else {
                FOR_UNROLL (int r = 0; r < RPL; r++) {
                    float4 acc = 0.0f;
                    FOR_UNROLL (int c = 0; c < NC; c++) {
                        ls[r][c] *= cge;
                        acc += ls[r][c]*kc[c];
                    }
                    sk[r] = (acc.x + acc.y) + (acc.z + acc.w);
                }
            }

            // one butterfly over the LPR lanes of the group: s_k(t) and y(t-1) together
            FOR_UNROLL (int l = 0; l < LLPR; l++) {
                FOR_UNROLL (int r = 0; r < RPL; r++) {
                    sk[r] += simd_shuffle_xor(sk[r], (ushort) (1 << l));
                    yp[r] += simd_shuffle_xor(yp[r], (ushort) (1 << l));
                }
            }

            if (t > 0 && sub < RPL) {
                float yo = yp[0];
                FOR_UNROLL (int r = 1; r < RPL; r++) { yo = sub == r ? yp[r] : yo; } // static select
                dst_attn[(uint) (t - 1)*H*SV + sub] = yo*scale;
            }

            FOR_UNROLL (int r = 0; r < RPL; r++) {
                const float d = (cv[r] - sk[r])*cb;
                float4 acc = 0.0f;
                FOR_UNROLL (int c = 0; c < NC; c++) {
                    ls[r][c] += kc[c]*d;
                    acc += ls[r][c]*qc[c];
                }
                yp[r] = (acc.x + acc.y) + (acc.z + acc.w);
            }

            if (KS > 1) {
                const int slot = T - 1 - t; // >= 0
                if (slot < KS) {
                    device float4 * ds4 = (device float4 *) (state_out + (uint) slot*slot_stride + state_base);
                    FOR_UNROLL (int r = 0; r < RPL; r++) {
                        FOR_UNROLL (int c = 0; c < NC; c++) {
                            ds4[r*NK4 + c*LPR + sub] = ls[r][c];
                        }
                    }
                }
            }
        }

        // 3. stage token t+1 (buffer (t+1)&1 was last read in token t-1, closed by the previous barrier)
        if (tn < T) {
            const int nxt = (tn & 1)*NBUF;
            FOR_UNROLL (int p = 0; p < NPF; p++) {
                const int i = lane + 32*p;
                if (i < 2*NK4) {
                    shmem[nxt + i] = pf[p];
                }
            }
            if (kda) {
                if (lane < NK4) {
                    shmem[nxt + 2*NK4 + lane] = exp(pfg);
                }
            } else {
                cge = exp(pgs);
            }
            FOR_UNROLL (int r = 0; r < RPL; r++) { cv[r] = pv[r]; }
            cb = pb;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // y of the last token
    FOR_UNROLL (int l = 0; l < LLPR; l++) {
        FOR_UNROLL (int r = 0; r < RPL; r++) {
            yp[r] += simd_shuffle_xor(yp[r], (ushort) (1 << l));
        }
    }
    if (sub < RPL) {
        float yo = yp[0];
        FOR_UNROLL (int r = 1; r < RPL; r++) { yo = sub == r ? yp[r] : yo; }
        dst_attn[(uint) (T - 1)*H*SV + sub] = yo*scale;
    }

    if (KS == 1) {
        device float4 * ds4 = (device float4 *) (state_out + state_base);
        FOR_UNROLL (int r = 0; r < RPL; r++) {
            FOR_UNROLL (int c = 0; c < NC; c++) {
                ds4[r*NK4 + c*LPR + sub] = ls[r][c];
            }
        }
    }
}

typedef decltype(kernel_gated_delta_net_rb_impl<128, 8, 2>) kernel_gated_delta_net_rb_t;

template [[host_name("kernel_gated_delta_net_rb_f32_32_8_1")]]  kernel kernel_gated_delta_net_rb_t kernel_gated_delta_net_rb_impl< 32, 8, 1>;
template [[host_name("kernel_gated_delta_net_rb_f32_32_8_2")]]  kernel kernel_gated_delta_net_rb_t kernel_gated_delta_net_rb_impl< 32, 8, 2>;
template [[host_name("kernel_gated_delta_net_rb_f32_32_4_2")]]  kernel kernel_gated_delta_net_rb_t kernel_gated_delta_net_rb_impl< 32, 4, 2>;
template [[host_name("kernel_gated_delta_net_rb_f32_64_8_1")]]  kernel kernel_gated_delta_net_rb_t kernel_gated_delta_net_rb_impl< 64, 8, 1>;
template [[host_name("kernel_gated_delta_net_rb_f32_64_8_2")]]  kernel kernel_gated_delta_net_rb_t kernel_gated_delta_net_rb_impl< 64, 8, 2>;
template [[host_name("kernel_gated_delta_net_rb_f32_64_4_2")]]  kernel kernel_gated_delta_net_rb_t kernel_gated_delta_net_rb_impl< 64, 4, 2>;
template [[host_name("kernel_gated_delta_net_rb_f32_128_8_1")]] kernel kernel_gated_delta_net_rb_t kernel_gated_delta_net_rb_impl<128, 8, 1>;
template [[host_name("kernel_gated_delta_net_rb_f32_128_8_2")]] kernel kernel_gated_delta_net_rb_t kernel_gated_delta_net_rb_impl<128, 8, 2>;
template [[host_name("kernel_gated_delta_net_rb_f32_128_4_2")]] kernel kernel_gated_delta_net_rb_t kernel_gated_delta_net_rb_impl<128, 4, 2>;
