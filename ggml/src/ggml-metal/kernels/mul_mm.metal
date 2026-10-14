#include "common.h"
#include "dequantize.h"

constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];
constant short FC_mul_mm_ne12  [[function_constant(FC_MUL_MM + 2)]];
constant short FC_mul_mm_ne13  [[function_constant(FC_MUL_MM + 3)]];
constant short FC_mul_mm_r2    [[function_constant(FC_MUL_MM + 4)]];
constant short FC_mul_mm_r3    [[function_constant(FC_MUL_MM + 5)]];
constant bool FC_mul_mm_id_amax [[function_constant(FC_MUL_MM + 6)]];
constant bool FC_mul_mm_wc_early [[function_constant(FC_MUL_MM + 7)]];
constant short FC_mul_mm_wc_ks   [[function_constant(FC_MUL_MM + 8)]]; // kernel_mul_mm_wc: K splits (1 = off)
constant bool  FC_mul_mm_wc_dq2  [[function_constant(FC_MUL_MM + 9)]]; // kernel_mul_mm_wc: dq2 A loaders (mm_wc_dq2_row, mm_wc_dq2_load/_dec)

// each block_q contains 16*nl weights
#ifdef GGML_METAL_HAS_TENSOR
template<
    typename SA, typename SA_4x4, typename SA_8x8,
    typename SB, typename SB_2x4, typename SB_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    // Matrix dimensions: A(M,K) x B(K,N) -> C(M,N)
    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    // Batch dimension handling
    const int im = tgpig.z;
    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    // Batch offsets for srcA and srcB
    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;

    // Tile dimensions
    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;
    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;

    // Tile offsets in output matrix
    const int ra = tgpig.y * NRA;
    const int rb = tgpig.x * NRB;

    // Threadgroup memory for dequantized A tile only
    threadgroup SA * sa = (threadgroup SA *)(shmem);

    // Work-item count for A loading
    constexpr int A_WORK_ITEMS = NRA * N_MM_NK;
    constexpr int NUM_THREADS = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;

    // tA wraps threadgroup memory
    auto tA = tensor(sa, dextents<int32_t, 2>(N_MM_NK_TOTAL, NRA));

    // tB wraps device memory directly
    device T1 * ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11 / sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    // Configure matmul operation
    // note: K is dynamic_extent (clamped to the valid range in PHASE 2), since a static
    //       N_MM_NK_TOTAL K tile would read src1 out of bounds when K % N_MM_NK_TOTAL != 0
    // ref: https://github.com/ggml-org/llama.cpp/pull/27064
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, static_cast<int>(dynamic_extent), false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    // Accumulate partial results over K dimension
    for (int loop_k = 0; loop_k < K; loop_k += N_MM_NK_TOTAL) {
        // === PHASE 1: Dequantization of A into threadgroup memory ===
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int row = work / N_MM_NK;
            const int k_chunk = work % N_MM_NK;
            const int k_pos = loop_k + k_chunk * 16;
            const short k_base = k_chunk * 16;

            // Bounds check: skip device read if row is out of matrix bounds
            if (ra + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    // Element-wise reads when K is not aligned (nb01 not aligned for half4x4/float4x4).
                    // MSL spec Table 2.5: half4x4 requires 8-byte alignment. When K is odd,
                    // nb01 = K*2 is not 8-byte aligned, so odd-row pointers are misaligned.
                    // Mirrors the legacy kernel's existing guard.
                    device const T0 * row_ptr = (device const T0 *)(srcA + args.nb01 * (ra + row) + offset0);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? (SA) row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos / (16 * nl);
                    const short il = (k_pos / 16) % nl;

                    device const block_q * row_ptr = (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        // Zero-pad A for K positions beyond valid range (handles partial K iterations)
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                // Zero-pad rows beyond matrix bounds
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row * N_MM_NK_TOTAL + (k_base + i)] = (SA)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 2: Tensor matmul ===
        // Clamp the K extent of both operand tensors to the remaining valid K range so
        // the dynamic-K op never reads past the K extent of src1 (or the staged A tile).
        const int kExt = min(N_MM_NK_TOTAL, K - loop_k);

        auto tAv = tensor(sa, dextents<int32_t, 2>(kExt, NRA), array<int, 2>({1, N_MM_NK_TOTAL}));
        auto tBv = tensor(ptrB + loop_k + rb * strideB, dextents<int32_t, 2>(kExt, N - rb), array<int, 2>({1, strideB}));

        mm.run(tBv, tAv, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result tile to output matrix (with batch offset)
    // cT.store handles bounds checking via tD's extents (M, N)
    device float * dstBatch = (device float *)dst + im * N * M;

    auto tD = tensor(dstBatch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    cT.store(tD.slice(ra, rb));
}

#else

template<
    typename S0, typename S0_4x4, typename S0_8x8,
    typename S1, typename S1_2x4, typename S1_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

#endif // GGML_METAL_HAS_TENSOR

//
// shared-memory register-blocked tiled matrix-matrix multiplication
//
// third mul_mm path for GPUs that have neither simdgroup-matrix
// (simdgroup_float8x8) nor the Metal tensor API. uses only threadgroup
// memory, barriers and scalar/vector FMA, so it always compiles and runs.
//
// C(M,N) = A(M,K) x B(N,K)^T. output tile is BM x BN, K walked in BK chunks.
// each thread owns a TM x TN micro-tile and accumulates it in registers.
// A (the weights) is dequantized ONCE per K-tile into threadgroup memory and
// reused across all BN columns of the tile. staging and the inner product
// use float4 loads.
//
// shared layout is K-major so the inner product reads are contiguous float4:
//   As[BK][BM] : As[kk*BM + m]
//   Bs[BK][BN] : Bs[kk*BN + n]
// stage one K-tile of A (dequantized weights) and B (src1) into half threadgroup
// memory. shared by the prologue and the prefetch of the double-buffered kernel.
template<
    typename T0, typename T0_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread T0_4x4 &),
    typename T1, typename T1_4, short BM, short BN, short BK, short NTHREAD>
void mm_tiled_stage(
        threadgroup half * As,
        threadgroup half * Bs,
        device const char * src0,
        device const T1 * ptrB,
        uint64_t nb01,
        uint64_t offset0,
        int strideB,
        int ra, int rb, int M, int N, int K, int k0,
        ushort tiitg) {
    constexpr short A_ITEMS = BM*(BK/16);
    constexpr short B_ITEMS = BN*(BK/4);

    for (short it = tiitg; it < A_ITEMS; it += NTHREAD) {
        const short m    = it % BM;
        const short sb   = it / BM;
        const int   kpos = k0 + sb*16;

        if (ra + m < M) {
            if (is_same<T0_4x4, block_q>::value) {
                device const T0 * row_ptr = (device const T0 *)(src0 + nb01*(ra + m) + offset0);
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    As[(sb*16 + i)*BM + m] = (kpos + i < K) ? (half) row_ptr[kpos + i] : (half) 0;
                }
            } else if (kpos < K) {
                const int   block_idx = kpos / (16*nl);
                const short il        = (kpos/16) % nl;
                device const block_q * row_ptr = (device const block_q *)(src0 + nb01*(ra + m) + offset0);
                T0_4x4 temp_a;
                dequantize_func(row_ptr + block_idx, il, temp_a);
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    As[(sb*16 + i)*BM + m] = (kpos + i < K) ? (half) temp_a[i/4][i%4] : (half) 0;
                }
            } else {
                // past the end of the row: do not read the next block
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    As[(sb*16 + i)*BM + m] = (half) 0;
                }
            }
        } else {
            FOR_UNROLL (short i = 0; i < 16; i++) {
                As[(sb*16 + i)*BM + m] = (half) 0;
            }
        }
    }

    for (short it = tiitg; it < B_ITEMS; it += NTHREAD) {
        const short n     = it / (BK/4);
        const short kbase = (it % (BK/4))*4;
        const int   gn    = rb + n;
        const int   kp    = k0 + kbase;

        half4 vh;
        if (gn < N && kp + 3 < K) {
            vh = (half4)(*(device const T1_4 *)(ptrB + (size_t) gn*strideB + kp));
        } else {
            vh.x = (gn < N && kp + 0 < K) ? (half) ptrB[(size_t) gn*strideB + kp + 0] : (half) 0;
            vh.y = (gn < N && kp + 1 < K) ? (half) ptrB[(size_t) gn*strideB + kp + 1] : (half) 0;
            vh.z = (gn < N && kp + 2 < K) ? (half) ptrB[(size_t) gn*strideB + kp + 2] : (half) 0;
            vh.w = (gn < N && kp + 3 < K) ? (half) ptrB[(size_t) gn*strideB + kp + 3] : (half) 0;
        }
        Bs[(kbase + 0)*BN + n] = vh.x;
        Bs[(kbase + 1)*BN + n] = vh.y;
        Bs[(kbase + 2)*BN + n] = vh.z;
        Bs[(kbase + 3)*BN + n] = vh.w;
    }
}

template<
    typename T0,       // src0 scalar for the plain f16/f32 path
    typename T0_4x4,   // 4x4 register produced by dequantize_func
    typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread T0_4x4 &),
    typename T1,       // src1 scalar
    typename T1_4>     // src1 4-vector (float4 for f32 src1, half4 for f16 src1)
kernel void kernel_mul_mm_tiled(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr short BM = 128;
    constexpr short BN = 128;
    constexpr short BK = 64;
    constexpr short TM = 8;
    constexpr short TN = 8;

    constexpr short NTX     = BN/TN;      // thread columns (N)
    constexpr short NTY     = BM/TM;      // thread rows (M)
    constexpr short NTHREAD = NTX*NTY;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    const int im  = tgpig.z;
    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;

    const int ra = tgpig.y*BM; // first output row (M) of this tile
    const int rb = tgpig.x*BN; // first output col (N) of this tile

    // single-buffered half shared memory
    threadgroup half * As = (threadgroup half *) shmem;
    threadgroup half * Bs = (threadgroup half *) shmem + BK*BM;

    device const T1 * ptrB = (device const T1 *)(src1 + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11/sizeof(T1);

    const short tc = tiitg % NTX; // this thread's column group (N)
    const short tr = tiitg / NTX; // this thread's row group (M)

    // FP32 master accumulator; half partials are folded into it every K-tile
    float4 acc[TM][TN/4];
    FOR_UNROLL (short i = 0; i < TM; i++) {
        FOR_UNROLL (short j = 0; j < TN/4; j++) {
            acc[i][j] = float4(0.0f);
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        mm_tiled_stage<T0, T0_4x4, block_q, nl, dequantize_func, T1, T1_4, BM, BN, BK, NTHREAD>(
            As, Bs, src0, ptrB, args.nb01, offset0, strideB, ra, rb, M, N, K, k0, tiitg);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // packed-fp16 multiply, accumulate in half across this BK chunk
        half4 hacc[TM][TN/4];
        FOR_UNROLL (short i = 0; i < TM; i++) {
            FOR_UNROLL (short j = 0; j < TN/4; j++) {
                hacc[i][j] = half4(0.0h);
            }
        }

        FOR_UNROLL (short kk = 0; kk < BK; kk++) {
            half4 a_reg[TM/4];
            half4 b_reg[TN/4];

            FOR_UNROLL (short i = 0; i < TM/4; i++) {
                a_reg[i] = *((threadgroup half4 *)(As + kk*BM + tr*TM) + i);
            }
            FOR_UNROLL (short j = 0; j < TN/4; j++) {
                b_reg[j] = *((threadgroup half4 *)(Bs + kk*BN + tc*TN) + j);
            }
            FOR_UNROLL (short i = 0; i < TM; i++) {
                const half a = a_reg[i/4][i%4];
                FOR_UNROLL (short j = 0; j < TN/4; j++) {
                    hacc[i][j] += a*b_reg[j];
                }
            }
        }

        // fold the half partials (BK products) into the FP32 accumulator
        FOR_UNROLL (short i = 0; i < TM; i++) {
            FOR_UNROLL (short j = 0; j < TN/4; j++) {
                acc[i][j] += float4(hacc[i][j]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float * dst_f32 = (device float *) dst + (size_t) im*N*M;

    FOR_UNROLL (short i = 0; i < TM; i++) {
        const int m = ra + tr*TM + i;
        if (m < M) {
            FOR_UNROLL (short j = 0; j < TN/4; j++) {
                const int n0 = rb + tc*TN + 4*j;
                const float4 r = acc[i][j];
                if (n0 + 0 < N) dst_f32[(size_t)(n0 + 0)*M + m] = r.x;
                if (n0 + 1 < N) dst_f32[(size_t)(n0 + 1)*M + m] = r.y;
                if (n0 + 2 < N) dst_f32[(size_t)(n0 + 2)*M + m] = r.z;
                if (n0 + 3 < N) dst_f32[(size_t)(n0 + 3)*M + m] = r.w;
            }
        }
    }
}

typedef decltype(kernel_mul_mm_tiled<half, half4x4, half4x4, 1, dequantize_f16, float, float4>) mul_mm_tiled_t;

template [[host_name("kernel_mul_mm_tiled_f32_f32")]]   kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, float4x4,   1,     dequantize_f32,   float, float4>;
template [[host_name("kernel_mul_mm_tiled_f16_f32")]]   kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, half4x4,    1,     dequantize_f16,   float, float4>;
template [[host_name("kernel_mul_mm_tiled_q4_0_f32")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q4_0,  2,     dequantize_q4_0,  float, float4>;
template [[host_name("kernel_mul_mm_tiled_q4_1_f32")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q4_1,  2,     dequantize_q4_1,  float, float4>;
template [[host_name("kernel_mul_mm_tiled_q5_0_f32")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q5_0,  2,     dequantize_q5_0,  float, float4>;
template [[host_name("kernel_mul_mm_tiled_q5_1_f32")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q5_1,  2,     dequantize_q5_1,  float, float4>;
template [[host_name("kernel_mul_mm_tiled_q8_0_f32")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q8_0,  2,     dequantize_q8_0,  float, float4>;
template [[host_name("kernel_mul_mm_tiled_q2_K_f32")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q2_K,  QK_NL, dequantize_q2_K,  float, float4>;
template [[host_name("kernel_mul_mm_tiled_q3_K_f32")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q3_K,  QK_NL, dequantize_q3_K,  float, float4>;
template [[host_name("kernel_mul_mm_tiled_q4_K_f32")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q4_K,  QK_NL, dequantize_q4_K,  float, float4>;
template [[host_name("kernel_mul_mm_tiled_q5_K_f32")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q5_K,  QK_NL, dequantize_q5_K,  float, float4>;
template [[host_name("kernel_mul_mm_tiled_q6_K_f32")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q6_K,  QK_NL, dequantize_q6_K,  float, float4>;
template [[host_name("kernel_mul_mm_tiled_iq3_s_f32")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq3_s, QK_NL, dequantize_iq3_s, float, float4>;
template [[host_name("kernel_mul_mm_tiled_mxfp4_f32")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_mxfp4,   2,     dequantize_mxfp4,   float, float4>;
template [[host_name("kernel_mul_mm_tiled_iq2_xxs_f32")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float4>;
template [[host_name("kernel_mul_mm_tiled_iq2_xs_f32")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, float4>;
template [[host_name("kernel_mul_mm_tiled_iq2_s_f32")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, float4>;
template [[host_name("kernel_mul_mm_tiled_iq3_xxs_f32")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float4>;
template [[host_name("kernel_mul_mm_tiled_iq1_s_f32")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, float4>;
template [[host_name("kernel_mul_mm_tiled_iq1_m_f32")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, float4>;
template [[host_name("kernel_mul_mm_tiled_iq4_nl_f32")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq4_nl,  2,     dequantize_iq4_nl,  float, float4>;
template [[host_name("kernel_mul_mm_tiled_iq4_xs_f32")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, float4>;

template [[host_name("kernel_mul_mm_tiled_f32_f16")]]   kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, float4x4,   1,     dequantize_f32,   half, half4>;
template [[host_name("kernel_mul_mm_tiled_f16_f16")]]   kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, half4x4,    1,     dequantize_f16,   half, half4>;
template [[host_name("kernel_mul_mm_tiled_q4_0_f16")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q4_0,  2,     dequantize_q4_0,  half, half4>;
template [[host_name("kernel_mul_mm_tiled_q4_1_f16")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q4_1,  2,     dequantize_q4_1,  half, half4>;
template [[host_name("kernel_mul_mm_tiled_q5_0_f16")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q5_0,  2,     dequantize_q5_0,  half, half4>;
template [[host_name("kernel_mul_mm_tiled_q5_1_f16")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q5_1,  2,     dequantize_q5_1,  half, half4>;
template [[host_name("kernel_mul_mm_tiled_q8_0_f16")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q8_0,  2,     dequantize_q8_0,  half, half4>;
template [[host_name("kernel_mul_mm_tiled_q2_K_f16")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q2_K,  QK_NL, dequantize_q2_K,  half, half4>;
template [[host_name("kernel_mul_mm_tiled_q3_K_f16")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q3_K,  QK_NL, dequantize_q3_K,  half, half4>;
template [[host_name("kernel_mul_mm_tiled_q4_K_f16")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q4_K,  QK_NL, dequantize_q4_K,  half, half4>;
template [[host_name("kernel_mul_mm_tiled_q5_K_f16")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q5_K,  QK_NL, dequantize_q5_K,  half, half4>;
template [[host_name("kernel_mul_mm_tiled_q6_K_f16")]]  kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_q6_K,  QK_NL, dequantize_q6_K,  half, half4>;
template [[host_name("kernel_mul_mm_tiled_iq3_s_f16")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq3_s, QK_NL, dequantize_iq3_s, half, half4>;
template [[host_name("kernel_mul_mm_tiled_mxfp4_f16")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_mxfp4,   2,     dequantize_mxfp4,   half, half4>;
template [[host_name("kernel_mul_mm_tiled_iq2_xxs_f16")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, half, half4>;
template [[host_name("kernel_mul_mm_tiled_iq2_xs_f16")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  half, half4>;
template [[host_name("kernel_mul_mm_tiled_iq2_s_f16")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq2_s,   QK_NL, dequantize_iq2_s,   half, half4>;
template [[host_name("kernel_mul_mm_tiled_iq3_xxs_f16")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, half, half4>;
template [[host_name("kernel_mul_mm_tiled_iq1_s_f16")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq1_s,   QK_NL, dequantize_iq1_s,   half, half4>;
template [[host_name("kernel_mul_mm_tiled_iq1_m_f16")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq1_m,   QK_NL, dequantize_iq1_m,   half, half4>;
template [[host_name("kernel_mul_mm_tiled_iq4_nl_f16")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq4_nl,  2,     dequantize_iq4_nl,  half, half4>;
template [[host_name("kernel_mul_mm_tiled_iq4_xs_f16")]] kernel mul_mm_tiled_t kernel_mul_mm_tiled<half, half4x4, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  half, half4>;

//
// kernel_mul_mm_wc: kernel_mul_mm_tiled with a bank-conflict-free threadgroup data path (default; GGML_METAL_MM_WC=0 selects tiled)
//
// same 128x128 tile, BK = 64, 256 threads, 8x8 micro-tile, packed-half FMA, FP32 fold per chunk, grid and numerics
// threadgroup memory holds K-major packed half2 words, 64 words per k-row:
//   As[k][w]  = (A[w][k], A[w+64][k])   at word w (a wave-load touches 32 consecutive rows); with dq2 at word w ^ (((k >> 3) & 3) << 3)
//   Bs[k][np] = (B[2np][k], B[2np+1][k]) at word np ^ (((k >> 3) & 3) << 3)
// staging stores are full words without bank conflicts (tiled: every B store is a 16-way conflict)
// each inner-loop operand is one 16-byte read

// 16 values of row `row` at k = kpos..kpos+15, zero outside the matrix
template<
    typename T0, typename T0_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread T0_4x4 &)>
T0_4x4 mm_wc_load_a(
        device const char * src0,
        uint64_t nb01,
        uint64_t offset0,
        int row, int M, int K, int kpos) {
    T0_4x4 r;

    if (row < M && kpos < K) {
        if (is_same<T0_4x4, block_q>::value) {
            device const T0 * p = (device const T0 *)(src0 + nb01*row + offset0) + kpos;
            FOR_UNROLL (int i = 0; i < 16; i++) {
                r[i/4][i%4] = (kpos + i < K) ? (half) p[i] : (half) 0;
            }
        } else {
            device const block_q * p = (device const block_q *)(src0 + nb01*row + offset0);
            dequantize_func(p + kpos/(16*nl), (short)((kpos/16) % nl), r);
            if (kpos + 16 > K) {
                FOR_UNROLL (int i = 0; i < 16; i++) {
                    r[i/4][i%4] = (kpos + i < K) ? r[i/4][i%4] : (half) 0;
                }
            }
        }
    } else {
        FOR_UNROLL (int i = 0; i < 16; i++) {
            r[i/4][i%4] = (half) 0;
        }
    }

    return r;
}

// dq2 A loaders (FC_mul_mm_wc_dq2): the quad of lanes aks = 0..3 of a row pair takes the 64-weight chunk c of a block,
// lane aks the weights l = 8*aks..8*aks+7 of both fields of the chunk, so it reads 8 quant bytes and the scales once.
// v[0], v[1] go to k = 8*aks + 0..7 of the chunk, v[2], v[3] to k = 32 + 8*aks + 0..7.
// after the f16 convert the values are the same as from dequantize_q3_K / dequantize_q4_K
template<typename block_q>
void mm_wc_dq2_row(device const block_q * xb, int c, int aks, thread float4 * v) {
    FOR_UNROLL (int i = 0; i < 4; i++) {
        v[i] = float4(0.0f);
    }
}

// q3_K chunk c: half n = c/2 of the block, fields j = 2*(c%2) + {0, 1}, weight 128*n + 32*j + l (l = 0..31)
// x = d*(s - 32)*(u - 4), u = q2 | hbit << 2, s = 6-bit scale of sub-block 8*n + 2*j + l/16
// (all products are exact in f32, so for finite d the f16 result equals dequantize_q3_K's)
// blocks are 2-byte aligned: ushort loads only
void mm_wc_dq2_row(device const block_q3_K * xb, int c, int aks, thread float4 * v) {
    const int n  = c/2;
    const int p  = c%2;
    const int a2 = aks/2;

    device const ushort * h16 = (device const ushort *) xb->hmask + 4*aks;
    device const ushort * q16 = (device const ushort *) xb->qs + 16*n + 4*aks;
    device const ushort * s16 = (device const ushort *) xb->scales;

    const uint h0 = (uint) h16[0] | ((uint) h16[1] << 16);
    const uint h1 = (uint) h16[2] | ((uint) h16[3] << 16);
    const uint q0 = (uint) q16[0] | ((uint) q16[1] << 16);
    const uint q1 = (uint) q16[2] | ((uint) q16[3] << 16);

    // sub-blocks 8*n + 4*p + a2 (field 2*p) and + 2 (field 2*p + 1): low 4 bits in scales[4*p + a2] (+ 2), high 2 bits in scales[8 + a2] (+ 2)
    const uint sa = s16[2*p + 0];
    const uint sb = s16[2*p + 1];
    const uint sh = ((uint) s16[4] | ((uint) s16[5] << 16)) >> (8*a2 + 2*(2*n + p));

    const int s0 = (int) (((sa >> (8*a2 + 4*n)) & 0xF) | ((sh <<  4) & 0x30));
    const int s1 = (int) (((sb >> (8*a2 + 4*n)) & 0xF) | ((sh >> 12) & 0x30));

    const float d   = (float) xb->d;
    const float dl0 = d*(float) (s0 - 32);
    const float dl1 = d*(float) (s1 - 32);

    FOR_UNROLL (int jj = 0; jj < 2; jj++) {
        const int   j  = 2*p + jj;
        const int   hb = 4*n + j;
        const float dl = jj == 0 ? dl0 : dl1;

        const uint u0 = ((q0 >> (2*j)) & 0x03030303u) | (((h0 >> hb) << 2) & 0x04040404u);
        const uint u1 = ((q1 >> (2*j)) & 0x03030303u) | (((h1 >> hb) << 2) & 0x04040404u);

        v[2*jj + 0] = fma(float4(as_type<uchar4>(u0)), float4(dl), float4(-4.0f*dl));
        v[2*jj + 1] = fma(float4(as_type<uchar4>(u1)), float4(dl), float4(-4.0f*dl));
    }
}

// q4_K chunk c: weight 64*c + l = low nibble of qs[32*c + l] (sub-block 2*c), 64*c + 32 + l = high nibble (sub-block 2*c + 1)
// blocks are 16-byte aligned (the host checks the offset and the strides)
void mm_wc_dq2_row(device const block_q4_K * xb, int c, int aks, thread float4 * v) {
    const uint4 hdr = *((device const uint4 *) xb);
    const uint2 qq  = *((device const uint2 *) (xb->qs + 32*c + 8*aks));

    // get_scale_min_k4_just2(2*c, 0/1, scales): bytes 2*(c%2) + {0, 1} of each 4-byte group
    const int  sh = 16*(c%2);
    const uint y  = hdr.y >> sh;
    const uint z  = hdr.z >> sh;
    const uint w  = hdr.w >> sh;

    uint sc0, sc1, mn0, mn1;
    if (c < 2) {
        sc0 =  y       & 63;
        sc1 = (y >> 8) & 63;
        mn0 =  z       & 63;
        mn1 = (z >> 8) & 63;
    } else {
        sc0 = ( w        & 0xF) | (( y       & 0xC0) >> 2);
        sc1 = ((w >>  8) & 0xF) | (((y >> 8) & 0xC0) >> 2);
        mn0 = ((w >>  4) & 0xF) | (( z       & 0xC0) >> 2);
        mn1 = ((w >> 12) & 0xF) | (((z >> 8) & 0xC0) >> 2);
    }

    const half  dh  = as_type<half2>(hdr.x)[0];
    const float dm  = (float) as_type<half2>(hdr.x)[1];
    const float dl0 = (float) dh*(float) sc0;
    const float dl1 = (float) (dh/16.h)*(float) sc1;
    const float ml0 = dm*(float) mn0;
    const float ml1 = dm*(float) mn1;

    v[0] = fma(float4(as_type<uchar4>(qq.x & 0x0F0F0F0Fu)), float4(dl0), float4(-ml0));
    v[1] = fma(float4(as_type<uchar4>(qq.y & 0x0F0F0F0Fu)), float4(dl0), float4(-ml0));
    v[2] = fma(float4(as_type<uchar4>(qq.x & 0xF0F0F0F0u)), float4(dl1), float4(-ml1));
    v[3] = fma(float4(as_type<uchar4>(qq.y & 0xF0F0F0F0u)), float4(dl1), float4(-ml1));
}

// split dq2 loaders (q6_K, iq4_xs, iq3_s, iq3_xxs, iq2_s): mm_wc_dq2_load reads the raw words of a row, mm_wc_dq2_dec
// makes v[] as mm_wc_dq2_row does (both rows are loaded before either is decoded). iq3_s, iq3_xxs, iq2_s put the unsigned
// products in v[] and the sign bits in sg (bit i for the staged word wa[i]): the kernel flips the sign of the f16 value,
// as the -1 factor of dequantize_iq* does, also for a NaN
#define MM_WC_DQ2_NR 8

template<typename block_q>
void mm_wc_dq2_load(device const block_q * xb, int c, int aks, thread uint * r) {
}

template<typename block_q>
void mm_wc_dq2_dec(device const block_q *, const thread uint * r, int c, int aks, thread float4 * v, thread uint & sg) {
    FOR_UNROLL (int i = 0; i < 4; i++) {
        v[i] = float4(0.0f);
    }
    sg = 0;
}

// q6_K chunk c: half n = c/2, fields j = 2*(c%2) + {0, 1}, weight 128*n + 32*j + l (l = 0..31)
// q = ((ql[64*n + 32*(j%2) + l] >> 4*(j/2)) & 0xF) | (((qh[32*n + l] >> 2*j) & 3) << 4), x = d*sc*(q - 32), sc = scales[8*n + 2*j + l/16]
// lane aks: l = 8*aks..8*aks + 7 of both fields
// (all products are exact in f32, so the f16 result equals dequantize_q6_K's)
// blocks are 2-byte aligned: ushort loads only
void mm_wc_dq2_load(device const block_q6_K * xb, int c, int aks, thread uint * r) {
    const int n = c/2;
    const int p = c%2;

    device const ushort * l16 = (device const ushort *) xb->ql + 32*n + 4*aks;
    device const ushort * h16 = (device const ushort *) xb->qh + 16*n + 4*aks;
    device const ushort * s16 = (device const ushort *) xb->scales + 4*n + 2*p;

    r[0] = (uint) l16[ 0] | ((uint) l16[ 1] << 16); // ql[64*n + 8*aks + 0..3]
    r[1] = (uint) l16[ 2] | ((uint) l16[ 3] << 16);
    r[2] = (uint) l16[16] | ((uint) l16[17] << 16); // ql[64*n + 32 + 8*aks + 0..3]
    r[3] = (uint) l16[18] | ((uint) l16[19] << 16);
    r[4] = (uint) h16[ 0] | ((uint) h16[ 1] << 16); // qh[32*n + 8*aks + 0..3]
    r[5] = (uint) h16[ 2] | ((uint) h16[ 3] << 16);
    r[6] = (uint) s16[ 0] | ((uint) s16[ 1] << 16); // scales[8*n + 4*p + 0..3]
    r[7] = as_type<ushort>(xb->d);
}

void mm_wc_dq2_dec(device const block_q6_K *, const thread uint * r, int c, int aks, thread float4 * v, thread uint & sg) {
    const int sh = 4*(c%2);

    // int8 scales[8*n + 4*p + aks/2] (field 2*p) and scales[8*n + 4*p + 2 + aks/2] (field 2*p + 1)
    const float d   = (float) as_type<half>((ushort) r[7]);
    const float dl0 = d*(float) (((int) (r[6] << (24 - 8*(aks/2)))) >> 24);
    const float dl1 = d*(float) (((int) (r[6] << ( 8 - 8*(aks/2)))) >> 24);

    const uint u0 = ((r[0] >> sh) & 0x0F0F0F0Fu) | (((r[4] >> sh) << 4) & 0x30303030u);
    const uint u1 = ((r[1] >> sh) & 0x0F0F0F0Fu) | (((r[5] >> sh) << 4) & 0x30303030u);
    const uint u2 = ((r[2] >> sh) & 0x0F0F0F0Fu) | (((r[4] >> sh) << 2) & 0x30303030u);
    const uint u3 = ((r[3] >> sh) & 0x0F0F0F0Fu) | (((r[5] >> sh) << 2) & 0x30303030u);

    v[0] = fma(float4(as_type<uchar4>(u0)), float4(dl0), float4(-32.0f*dl0));
    v[1] = fma(float4(as_type<uchar4>(u1)), float4(dl0), float4(-32.0f*dl0));
    v[2] = fma(float4(as_type<uchar4>(u2)), float4(dl1), float4(-32.0f*dl1));
    v[3] = fma(float4(as_type<uchar4>(u3)), float4(dl1), float4(-32.0f*dl1));
    sg = 0;
}

// iq4_xs chunk c: sub-blocks 2*c (k < 32) and 2*c + 1, weight 32*ib + l = nibble l/16 of qs[16*ib + l%16]
// lane aks: bytes qs[32*c + 8*(aks%2) + 0..7] and qs[32*c + 16 + 8*(aks%2) + 0..7], nibble aks/2
// x = d*(ls - 32)*kvalues_iq4nl[q]: the products are exact in f32, so the f16 result equals dequantize_iq4_xs's
// blocks are 8-byte aligned (the host checks the offset and the strides)
void mm_wc_dq2_load(device const block_iq4_xs * xb, int c, int aks, thread uint * r) {
    const uint2 hdr = *((device const uint2 *) xb); // d, scales_h, scales_l[0..3]
    device const uint2 * q2 = (device const uint2 *) (xb->qs + 32*c + 8*(aks%2));
    const uint2 qa = q2[0];
    const uint2 qb = q2[2];

    r[0] = hdr.x;
    r[1] = hdr.y;
    r[2] = qa.x;
    r[3] = qa.y;
    r[4] = qb.x;
    r[5] = qb.y;
}

void mm_wc_dq2_dec(device const block_iq4_xs *, const thread uint * r, int c, int aks, thread float4 * v, thread uint & sg) {
    const uint ls0 = ((r[1] >> (8*c    )) & 0xF) | (((r[0] >> (16 + 4*c)) & 3) << 4);
    const uint ls1 = ((r[1] >> (8*c + 4)) & 0xF) | (((r[0] >> (18 + 4*c)) & 3) << 4);

    const float d   = (float) as_type<half2>(r[0])[0];
    const float dl0 = d*(float) ((int) ls0 - 32);
    const float dl1 = d*(float) ((int) ls1 - 32);

    const int sh = 4*(aks/2);
    FOR_UNROLL (int i = 0; i < 4; i++) {
        const uchar4 q  = as_type<uchar4>((r[2 + i] >> sh) & 0x0F0F0F0Fu);
        const float  dl = i < 2 ? dl0 : dl1;
        v[i] = dl*float4(kvalues_iq4nl_f[q[0]], kvalues_iq4nl_f[q[1]], kvalues_iq4nl_f[q[2]], kvalues_iq4nl_f[q[3]]);
    }
    sg = 0;
}

// iq3_s chunk c: sub-blocks 2*c (k < 32) and 2*c + 1, weight 32*ib + l = byte l%4 of iq3s_grid[qs[8*ib + l/4] | bit l/4 of qh[ib] << 8],
// negative if bit l%8 of signs[4*ib + l/8] is set, scale d*(1 + 2*nibble ib%2 of scales[ib/2])
// lane aks: grid entries 2*aks, 2*aks + 1 and sign byte aks of both sub-blocks
// (the products are exact in f32, so the f16 result equals dequantize_iq3_s's)
// blocks are 2-byte aligned: ushort loads only
void mm_wc_dq2_load(device const block_iq3_s * xb, int c, int aks, thread uint * r) {
    device const ushort * x16 = (device const ushort *) xb;

    r[0] = (uint) x16[ 1 + 8*c + aks]   | ((uint) x16[ 5 + 8*c + aks]   << 16); // qs[16*c + 2*aks + 0..1], qs[16*c + 8 + 2*aks + 0..1]
    r[1] = (uint) x16[33 + c]           | ((uint) x16[53 + c/2]         << 16); // qh[2*c..2*c + 1], scales[2*(c/2)..+1]
    r[2] = (uint) x16[37 + 4*c + aks/2] | ((uint) x16[39 + 4*c + aks/2] << 16); // signs[8*c + aks], signs[8*c + 4 + aks] in byte aks%2
    r[3] = as_type<ushort>(xb->d);
}

void mm_wc_dq2_dec(device const block_iq3_s *, const thread uint * r, int c, int aks, thread float4 * v, thread uint & sg) {
    const uint qh = r[1];
    const uint sc = r[1] >> (16 + 8*(c%2));
    const uint s  = r[2] >> (8*(aks%2)); // sign byte of sub-block 2*c in bits 0..7, of 2*c + 1 in bits 16..23

    const float d   = (float) as_type<half>((ushort) r[3]);
    const float dl0 = d*(float) (1 + 2*(int) ( sc       & 0xF));
    const float dl1 = d*(float) (1 + 2*(int) ((sc >> 4) & 0xF));

    FOR_UNROLL (int i = 0; i < 4; i++) {
        const uint  g  = iq3s_grid[((r[0] >> 8*i) & 0xFF) | (((qh >> (8*(i/2) + 2*aks + i%2)) & 1) << 8)];
        const float dl = i < 2 ? dl0 : dl1;
        v[i] = dl*float4(as_type<uchar4>(g));
    }
    sg = (s & 0xFF) | ((s >> 8) & 0xFF00);
}

// iq3_xxs chunk c: sub-blocks 2*c (k < 32) and 2*c + 1, weight 32*ib + l = byte l%4 of iq3xxs_grid[qs[8*ib + l/4]],
// sign bit l%8 of ksigns_iq2xs[bits 7*(l/8)..+6 of aux], scale d*(0.5 + aux >> 28)*0.5, aux = uint32 at qs[64 + 4*ib]
// lane aks: grid entries 2*aks, 2*aks + 1 and sign group aks of both sub-blocks
// (the products are exact in f32, so the f16 result equals dequantize_iq3_xxs's)
// blocks are 2-byte aligned: ushort loads only
void mm_wc_dq2_load(device const block_iq3_xxs * xb, int c, int aks, thread uint * r) {
    device const ushort * x16 = (device const ushort *) xb;

    r[0] = (uint) x16[ 1 + 8*c + aks] | ((uint) x16[ 5 + 8*c + aks] << 16); // qs[16*c + 2*aks + 0..1], qs[16*c + 8 + 2*aks + 0..1]
    r[1] = (uint) x16[33 + 4*c]       | ((uint) x16[34 + 4*c]       << 16); // aux of sub-block 2*c
    r[2] = (uint) x16[35 + 4*c]       | ((uint) x16[36 + 4*c]       << 16); // aux of sub-block 2*c + 1
    r[3] = as_type<ushort>(xb->d);
}

void mm_wc_dq2_dec(device const block_iq3_xxs *, const thread uint * r, int c, int aks, thread float4 * v, thread uint & sg) {
    const float d = (float) as_type<half>((ushort) r[3]);

    sg = 0;
    FOR_UNROLL (int h = 0; h < 2; h++) {
        const uint  aux = r[1 + h];
        const float dl  = d*(0.5f + (float) (aux >> 28))*0.5f;
        const uint  s7  = (aux >> (7*aks)) & 127;

        sg |= (s7 | ((popcount(s7) & 1) << 7)) << 8*h; // ksigns_iq2xs[s7]

        FOR_UNROLL (int i = 0; i < 2; i++) {
            v[2*h + i] = dl*float4(as_type<uchar4>(iq3xxs_grid[(r[0] >> (16*h + 8*i)) & 0xFF]));
        }
    }
}

// iq2_s chunk c: sub-blocks 2*c (k < 32) and 2*c + 1, weight 32*ib + l = byte l%8 of iq2s_grid[qs[4*ib + l/8] | ((qh[ib] >> 2*(l/8)) & 3) << 8],
// negative if bit l%8 of qs[32 + 4*ib + l/8] is set, scale d*(0.5 + nibble l/16 of scales[ib])*0.25
// lane aks: grid entry aks and sign byte aks of both sub-blocks
// (the products are exact in f32, so the f16 result equals dequantize_iq2_s's)
// blocks are 2-byte aligned: ushort loads only
void mm_wc_dq2_load(device const block_iq2_s * xb, int c, int aks, thread uint * r) {
    device const ushort * x16 = (device const ushort *) xb;

    r[0] = (uint) x16[ 1 + 4*c + aks/2] | ((uint) x16[ 3 + 4*c + aks/2] << 16); // qs[8*c + aks], qs[8*c + 4 + aks] in byte aks%2
    r[1] = (uint) x16[17 + 4*c + aks/2] | ((uint) x16[19 + 4*c + aks/2] << 16); // signs qs[32 + 8*c + aks], qs[32 + 8*c + 4 + aks]
    r[2] = (uint) x16[33 + c]           | ((uint) x16[37 + c]           << 16); // qh[2*c..2*c + 1], scales[2*c..2*c + 1]
    r[3] = as_type<ushort>(xb->d);
}

void mm_wc_dq2_dec(device const block_iq2_s *, const thread uint * r, int c, int aks, thread float4 * v, thread uint & sg) {
    const float d = (float) as_type<half>((ushort) r[3]);

    const uint q = r[0] >> (8*(aks%2));
    const uint s = r[1] >> (8*(aks%2));

    FOR_UNROLL (int h = 0; h < 2; h++) {
        const float dl = d*(0.5f + (float) ((r[2] >> (16 + 8*h + 4*(aks/2))) & 0xF))*0.25f;
        const uint2 g  = ((constant uint2 *) iq2s_grid)[((q >> 16*h) & 0xFF) | (((r[2] >> (8*h + 2*aks)) & 3) << 8)];

        v[2*h + 0] = dl*float4(as_type<uchar4>(g.x));
        v[2*h + 1] = dl*float4(as_type<uchar4>(g.y));
    }
    sg = (s & 0xFF) | ((s >> 8) & 0xFF00);
}

// k row in the chunk of the staged A word wa[i]
static inline int mm_wc_ka(bool dq2, int aks, int i) {
    return dq2 ? (i < 8 ? 8*aks + i : 24 + 8*aks + i) : 16*aks + i;
}

template<
    typename T0, typename T0_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread T0_4x4 &),
    typename T1, typename T1_4>
kernel void kernel_mul_mm_wc(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg_u [[thread_index_in_threadgroup]]) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int NW = 64; // packed words per k-row in As and Bs

    // all index math in int: short address math miscompiles on this backend
    const int tiitg = tiitg_u;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    // split-K (FC_mul_mm_wc_ks > 1, for grids with few tiles): grid z = batch*ks, each threadgroup sums
    // whole BK chunks of its K range and writes an FP32 partial plane [ks][N][M] per batch into dst
    // (a scratch buffer), the planes are added by kernel_mul_mm_wc_ksum
    // KS == 1 resolves at pipeline creation to the plain expressions (the generic ones cost ~2% there)
    const int KS  = FC_mul_mm_wc_ks;
    const int im  = KS == 1 ? (int) tgpig.z : (int) tgpig.z / KS;
    const int kz  = KS == 1 ? 0             : (int) tgpig.z % KS;
    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    // whole BK chunks per split, the first nch % KS splits get one more
    const int nch = (K + BK - 1)/BK;
    const int kb0 = KS == 1 ? 0 : (kz*(nch/KS) + min(kz, nch % KS))*BK;
    const int kb1 = KS == 1 ? K : min(K, kb0 + (nch/KS + (kz < nch % KS ? 1 : 0))*BK);

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;

    const int ra = tgpig.y*BM;
    const int rb = tgpig.x*BN;

    threadgroup uint * As = (threadgroup uint *) shmem;         // [BK][NW]
    threadgroup uint * Bs = (threadgroup uint *) shmem + BK*NW; // [BK][NW], swizzled

    device const T1 * ptrB = (device const T1 *)(src1 + args.nb12*i12 + args.nb13*i13);

    const int  strideB  = args.nb11/sizeof(T1);
    const bool b_vec_ok = strideB % 4 == 0;

    // compute map, as kernel_mul_mm_tiled: rows TM*tr..TM*tr+7, cols TN*tc..TN*tc+7
    const int tc = tiitg % (BN/TN);
    const int tr = tiitg / (BN/TN);

    // A staging: rows arp and arp+64 (one wave-load touches 32 consecutive rows), k = 16*aks..+15
    // dq2: the 4 lanes of a row pair are a quad (one wave-load touches 8 row pairs), k = 8*aks..+7 and 32 + 8*aks..+7,
    // and the As word column is XOR-swizzled by aks << 3 (key (k >> 3) & 3, as Bs) to keep the stores conflict-free
    const int arp = FC_mul_mm_wc_dq2 ? tiitg/4 : tiitg%64;
    const int aks = FC_mul_mm_wc_dq2 ? tiitg%4 : tiitg/64;
    const int asw = FC_mul_mm_wc_dq2 ? arp ^ (aks << 3) : arp;

    // B staging: cols 2*bnp and 2*bnp+1, k octets bko and bko+4 (one wave = 8 col pairs x 4 octets)
    const int lane = tiitg % 32;
    const int bnp  = 8*(tiitg/32) + lane%8;
    const int bko  = lane/8;
    const int bcol = bnp ^ (bko << 3); // swizzle key (k >> 3) & 3 is bko for both octets
    const int bn0  = rb + 2*bnp;

    float4 acc[TN][TM/4]; // [col][rows 4i..4i+3]
    FOR_UNROLL (int j = 0; j < TN; j++) {
        FOR_UNROLL (int i = 0; i < TM/4; i++) {
            acc[j][i] = float4(0.0f);
        }
    }

    // dq2 split loaders (all dq2 types but q3_K and q4_K) with the row pointers out of the loop
    // QK_K blocks, K % 256 == 0, so every chunk is whole. rows past M read row M - 1 (their outputs are not stored)
    constexpr bool dq2_sp = !is_same<block_q, block_q3_K>::value && !is_same<block_q, block_q4_K>::value;
    device const block_q * xr0 = (device const block_q *) (src0 + offset0 + args.nb01*min(ra + arp,      M - 1));
    device const block_q * xr1 = (device const block_q *) (src0 + offset0 + args.nb01*min(ra + arp + 64, M - 1));

    for (int k0 = kb0; k0 < kb1; k0 += BK) {
        if (!FC_mul_mm_wc_early && k0 > kb0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint wa[16];
        if (FC_mul_mm_wc_dq2 && dq2_sp) {
            uint r0[MM_WC_DQ2_NR];
            uint r1[MM_WC_DQ2_NR];
            mm_wc_dq2_load(xr0 + k0/QK_K, (k0/64) % 4, aks, r0);
            mm_wc_dq2_load(xr1 + k0/QK_K, (k0/64) % 4, aks, r1);
            float4 v0[4];
            float4 v1[4];
            uint sg0;
            uint sg1;
            mm_wc_dq2_dec(xr0, r0, (k0/64) % 4, aks, v0, sg0);
            mm_wc_dq2_dec(xr1, r1, (k0/64) % 4, aks, v1, sg1);
            const uint sg = sg0 | (sg1 << 16);
            FOR_UNROLL (int i = 0; i < 16; i++) {
                wa[i] = as_type<uint>(half2((half) v0[i/4][i%4], (half) v1[i/4][i%4])) ^ ((sg << (15 - i)) & 0x80008000u);
            }
        } else if (FC_mul_mm_wc_dq2) {
            // K-quants: K % 256 == 0, so every chunk is whole. rows past M read row M - 1 (their outputs are not stored)
            device const block_q * x0 = (device const block_q *) (src0 + offset0 + args.nb01*min(ra + arp,      M - 1)) + k0/QK_K;
            device const block_q * x1 = (device const block_q *) (src0 + offset0 + args.nb01*min(ra + arp + 64, M - 1)) + k0/QK_K;
            float4 v0[4];
            float4 v1[4];
            mm_wc_dq2_row(x0, (k0/64) % 4, aks, v0);
            mm_wc_dq2_row(x1, (k0/64) % 4, aks, v1);
            FOR_UNROLL (int i = 0; i < 16; i++) {
                wa[i] = as_type<uint>(half2((half) v0[i/4][i%4], (half) v1[i/4][i%4]));
            }
        } else {
            const int kpos = k0 + 16*aks;
            const T0_4x4 r0 = mm_wc_load_a<T0, T0_4x4, block_q, nl, dequantize_func>(src0, args.nb01, offset0, ra + arp,      M, K, kpos);
            const T0_4x4 r1 = mm_wc_load_a<T0, T0_4x4, block_q, nl, dequantize_func>(src0, args.nb01, offset0, ra + arp + 64, M, K, kpos);
            FOR_UNROLL (int i = 0; i < 16; i++) {
                wa[i] = as_type<uint>(half2(r0[i/4][i%4], r1[i/4][i%4]));
            }
        }

        // late staging: store A before loading B to keep fewer registers live
        if (!FC_mul_mm_wc_early) {
            FOR_UNROLL (int i = 0; i < 16; i++) {
                As[mm_wc_ka(FC_mul_mm_wc_dq2, aks, i)*NW + asw] = wa[i];
            }
        }

        uint wb[16];
        FOR_UNROLL (int t = 0; t < 2; t++) {
            const int kb = k0 + 8*bko + 32*t;

            half4 x0[2];
            half4 x1[2];
            if (b_vec_ok && bn0 + 1 < N && kb + 8 <= K) {
                device const T1_4 * p0 = (device const T1_4 *)(ptrB + (size_t)(bn0 + 0)*strideB + kb);
                device const T1_4 * p1 = (device const T1_4 *)(ptrB + (size_t)(bn0 + 1)*strideB + kb);
                x0[0] = (half4) p0[0];
                x0[1] = (half4) p0[1];
                x1[0] = (half4) p1[0];
                x1[1] = (half4) p1[1];
            } else {
                FOR_UNROLL (int c = 0; c < 8; c++) {
                    x0[c/4][c%4] = (bn0 + 0 < N && kb + c < K) ? (half) ptrB[(size_t)(bn0 + 0)*strideB + kb + c] : (half) 0;
                    x1[c/4][c%4] = (bn0 + 1 < N && kb + c < K) ? (half) ptrB[(size_t)(bn0 + 1)*strideB + kb + c] : (half) 0;
                }
            }

            FOR_UNROLL (int c = 0; c < 8; c++) {
                wb[8*t + c] = as_type<uint>(half2(x0[c/4][c%4], x1[c/4][c%4]));
            }
        }

        // early staging: the global loads above overlap the slower waves of the previous chunk
        if (FC_mul_mm_wc_early && k0 > kb0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // a wave writes 32 consecutive words (A; 32 distinct banks with dq2) or 32 distinct banks (B)
        if (FC_mul_mm_wc_early) {
            FOR_UNROLL (int i = 0; i < 16; i++) {
                As[mm_wc_ka(FC_mul_mm_wc_dq2, aks, i)*NW + asw] = wa[i];
            }
        }
        FOR_UNROLL (int t = 0; t < 2; t++) {
            FOR_UNROLL (int c = 0; c < 8; c++) {
                Bs[(8*bko + 32*t + c)*NW + bcol] = wb[8*t + c];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        half4 hacc[TN][TM/4]; // [col TN*tc+j][rows TM*tr+4i..+3]
        FOR_UNROLL (int j = 0; j < TN; j++) {
            FOR_UNROLL (int i = 0; i < TM/4; i++) {
                hacc[j][i] = half4(0.0h);
            }
        }

        FOR_UNROLL (int kk = 0; kk < BK; kk++) {
            const int xb = ((kk >> 3) & 3) << 3;

            const uint4 ua = *((threadgroup const uint4 *)(As + kk*NW + (FC_mul_mm_wc_dq2 ? (4*tr) ^ xb : 4*tr)));
            const uint4 ub = *((threadgroup const uint4 *)(Bs + kk*NW + ((4*tc) ^ xb)));

            const half4 a0 = as_type<half4>(ua.xy); // rows {4tr, 4tr+64, 4tr+1, 4tr+65}
            const half4 a1 = as_type<half4>(ua.zw); // rows {4tr+2, 4tr+66, 4tr+3, 4tr+67}
            const half4 b0 = as_type<half4>(ub.xy); // cols TN*tc+0..3
            const half4 b1 = as_type<half4>(ub.zw); // cols TN*tc+4..7

            FOR_UNROLL (int j = 0; j < 4; j++) {
                hacc[j + 0][0] = fma(half4(b0[j]), a0, hacc[j + 0][0]);
                hacc[j + 0][1] = fma(half4(b0[j]), a1, hacc[j + 0][1]);
                hacc[j + 4][0] = fma(half4(b1[j]), a0, hacc[j + 4][0]);
                hacc[j + 4][1] = fma(half4(b1[j]), a1, hacc[j + 4][1]);
            }
        }

        // fold the half partials (BK products) into FP32, as kernel_mul_mm_tiled
        FOR_UNROLL (int j = 0; j < TN; j++) {
            FOR_UNROLL (int i = 0; i < TM/4; i++) {
                acc[j][i] += float4(hacc[j][i]);
            }
        }
    }

    device float * dst_f32 = (device float *) dst + ((size_t) im*KS + kz)*N*M;

    const bool m_vec = M % 4 == 0;

    FOR_UNROLL (int j = 0; j < TN; j++) {
        const int n = rb + TN*tc + j;
        if (n < N) {
            device float * D = dst_f32 + (size_t) n*M;
            const float4 lo = float4(acc[j][0].x, acc[j][0].z, acc[j][1].x, acc[j][1].z); // rows 4tr..4tr+3
            const float4 hi = float4(acc[j][0].y, acc[j][0].w, acc[j][1].y, acc[j][1].w); // rows 4tr+64..+67
            FOR_UNROLL (int i = 0; i < 2; i++) {
                const int m = ra + 4*tr + 64*i;
                const float4 v = i == 0 ? lo : hi;
                if (m_vec && m < M) {
                    *((device float4 *)(D + m)) = v;
                } else {
                    FOR_UNROLL (int c = 0; c < 4; c++) {
                        if (m + c < M) {
                            D[m + c] = v[c];
                        }
                    }
                }
            }
        }
    }
}

typedef decltype(kernel_mul_mm_wc<half, half4x4, half4x4, 1, dequantize_f16, float, float4>) mul_mm_wc_t;

template [[host_name("kernel_mul_mm_wc_f32_f32")]]   kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, float4x4,    1,     dequantize_f32,   float, float4>;
template [[host_name("kernel_mul_mm_wc_f16_f32")]]   kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, half4x4,     1,     dequantize_f16,   float, float4>;
template [[host_name("kernel_mul_mm_wc_q4_0_f32")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q4_0,  2,     dequantize_q4_0,  float, float4>;
template [[host_name("kernel_mul_mm_wc_q4_1_f32")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q4_1,  2,     dequantize_q4_1,  float, float4>;
template [[host_name("kernel_mul_mm_wc_q5_0_f32")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q5_0,  2,     dequantize_q5_0,  float, float4>;
template [[host_name("kernel_mul_mm_wc_q5_1_f32")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q5_1,  2,     dequantize_q5_1,  float, float4>;
template [[host_name("kernel_mul_mm_wc_q8_0_f32")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q8_0,  2,     dequantize_q8_0,  float, float4>;
template [[host_name("kernel_mul_mm_wc_q2_K_f32")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q2_K,  QK_NL, dequantize_q2_K,  float, float4>;
template [[host_name("kernel_mul_mm_wc_q3_K_f32")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q3_K,  QK_NL, dequantize_q3_K,  float, float4>;
template [[host_name("kernel_mul_mm_wc_q4_K_f32")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q4_K,  QK_NL, dequantize_q4_K,  float, float4>;
template [[host_name("kernel_mul_mm_wc_q5_K_f32")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q5_K,  QK_NL, dequantize_q5_K,  float, float4>;
template [[host_name("kernel_mul_mm_wc_q6_K_f32")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q6_K,  QK_NL, dequantize_q6_K,  float, float4>;
template [[host_name("kernel_mul_mm_wc_iq3_s_f32")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq3_s, QK_NL, dequantize_iq3_s, float, float4>;
template [[host_name("kernel_mul_mm_wc_mxfp4_f32")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_mxfp4,   2,     dequantize_mxfp4,   float, float4>;
template [[host_name("kernel_mul_mm_wc_iq2_xxs_f32")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float4>;
template [[host_name("kernel_mul_mm_wc_iq2_xs_f32")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, float4>;
template [[host_name("kernel_mul_mm_wc_iq2_s_f32")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, float4>;
template [[host_name("kernel_mul_mm_wc_iq3_xxs_f32")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float4>;
template [[host_name("kernel_mul_mm_wc_iq1_s_f32")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, float4>;
template [[host_name("kernel_mul_mm_wc_iq1_m_f32")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, float4>;
template [[host_name("kernel_mul_mm_wc_iq4_nl_f32")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq4_nl,  2,     dequantize_iq4_nl,  float, float4>;
template [[host_name("kernel_mul_mm_wc_iq4_xs_f32")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, float4>;

template [[host_name("kernel_mul_mm_wc_f32_f16")]]   kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, float4x4,    1,     dequantize_f32,   half, half4>;
template [[host_name("kernel_mul_mm_wc_f16_f16")]]   kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, half4x4,     1,     dequantize_f16,   half, half4>;
template [[host_name("kernel_mul_mm_wc_q4_0_f16")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q4_0,  2,     dequantize_q4_0,  half, half4>;
template [[host_name("kernel_mul_mm_wc_q4_1_f16")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q4_1,  2,     dequantize_q4_1,  half, half4>;
template [[host_name("kernel_mul_mm_wc_q5_0_f16")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q5_0,  2,     dequantize_q5_0,  half, half4>;
template [[host_name("kernel_mul_mm_wc_q5_1_f16")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q5_1,  2,     dequantize_q5_1,  half, half4>;
template [[host_name("kernel_mul_mm_wc_q8_0_f16")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q8_0,  2,     dequantize_q8_0,  half, half4>;
template [[host_name("kernel_mul_mm_wc_q2_K_f16")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q2_K,  QK_NL, dequantize_q2_K,  half, half4>;
template [[host_name("kernel_mul_mm_wc_q3_K_f16")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q3_K,  QK_NL, dequantize_q3_K,  half, half4>;
template [[host_name("kernel_mul_mm_wc_q4_K_f16")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q4_K,  QK_NL, dequantize_q4_K,  half, half4>;
template [[host_name("kernel_mul_mm_wc_q5_K_f16")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q5_K,  QK_NL, dequantize_q5_K,  half, half4>;
template [[host_name("kernel_mul_mm_wc_q6_K_f16")]]  kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_q6_K,  QK_NL, dequantize_q6_K,  half, half4>;
template [[host_name("kernel_mul_mm_wc_iq3_s_f16")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq3_s, QK_NL, dequantize_iq3_s, half, half4>;
template [[host_name("kernel_mul_mm_wc_mxfp4_f16")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_mxfp4,   2,     dequantize_mxfp4,   half, half4>;
template [[host_name("kernel_mul_mm_wc_iq2_xxs_f16")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, half, half4>;
template [[host_name("kernel_mul_mm_wc_iq2_xs_f16")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  half, half4>;
template [[host_name("kernel_mul_mm_wc_iq2_s_f16")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq2_s,   QK_NL, dequantize_iq2_s,   half, half4>;
template [[host_name("kernel_mul_mm_wc_iq3_xxs_f16")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, half, half4>;
template [[host_name("kernel_mul_mm_wc_iq1_s_f16")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq1_s,   QK_NL, dequantize_iq1_s,   half, half4>;
template [[host_name("kernel_mul_mm_wc_iq1_m_f16")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq1_m,   QK_NL, dequantize_iq1_m,   half, half4>;
template [[host_name("kernel_mul_mm_wc_iq4_nl_f16")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq4_nl,  2,     dequantize_iq4_nl,  half, half4>;
template [[host_name("kernel_mul_mm_wc_iq4_xs_f16")]] kernel mul_mm_wc_t kernel_mul_mm_wc<half, half4x4, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  half, half4>;

// adds the FP32 partial planes of a split-K kernel_mul_mm_wc in a fixed order: dst[b][i] = sum_k src[b][k][i]
// args.x = elements per plane (N*M), args.y = number of splits. grid: (ceil(args.x/ntg), batch)
kernel void kernel_mul_mm_wc_ksum(
        constant uint2 & args,
        device const float * src,
        device       float * dst,
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint2 tpitg [[thread_position_in_threadgroup]],
        uint2 ntg   [[threads_per_threadgroup]]) {
    const uint n  = args.x;
    const uint ks = args.y;
    const uint i  = tgpig.x*ntg.x + tpitg.x;

    if (i >= n) {
        return;
    }

    device const float * s = src + (size_t) tgpig.y*ks*n + i;

    float sum = 0.0f;
    for (uint k = 0; k < ks; k++) {
        sum += s[(size_t) k*n];
    }

    dst[(size_t) tgpig.y*n + i] = sum;
}

// activation f32->f16 convert for the f16-activation tiled path (one pass, reused scratch)
kernel void kernel_mm_cvt_f32_f16(
        device const float * src [[buffer(0)]],
        device       half  * dst [[buffer(1)]],
        constant     uint  & ne  [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < ne) {
        dst[gid] = (half) src[gid];
    }
}

//
// MoE: scalar shared-memory register-blocked tiled mul_mm_id (expert matmul)
//
// same FP16 tiled accumulate as kernel_mul_mm_tiled, with expert routing:
//   - tgpig.z selects the expert; its weights live at src0 + im*nb02
//   - tgpig.x tiles the tokens routed to this expert (count = htpe[im])
//   - each B column (token) GATHERS its src1 row via hids; the result is
//     SCATTERED back to dst via hids (same mapping the stock mul_mm_id uses)
template<
    typename T0, typename T0_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread T0_4x4 &),
    typename T1>
kernel void kernel_mul_mm_id_tiled(
        constant ggml_metal_kargs_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        device const char * amax,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr short BM = 64;
    constexpr short BN = 64;
    constexpr short BK = 16;
    constexpr short TM = 8;
    constexpr short TN = 8;

    constexpr short NTX     = BN/TN;
    constexpr short NTY     = BM/TM;
    constexpr short NTHREAD = NTX*NTY;
    constexpr short A_ITEMS = BM*(BK/16);

    const int K = args.ne00;
    const int M = args.ne0;

    const int im = tgpig.z; // expert
    const int r0 = tgpig.y*BM; // output-feature (M) tile
    const int r1 = tgpig.x*BN; // token tile within this expert

    device const uint32_t * tpe_u32 = (device const uint32_t *) htpe;
    device const int32_t  * ids_i32 = (device const int32_t  *) hids;

    const int neh1 = (int) tpe_u32[im]; // tokens routed to this expert
    if (r1 >= neh1) {
        return;
    }

    const uint64_t offset0 = im*args.nb02; // this expert's weight matrix

    const float s1_inv   = FC_mul_mm_id_amax ? ((device const float *) amax)[0] : 1.0f;
    const float s1_scale = FC_mul_mm_id_amax ? ((device const float *) amax)[1] : 1.0f;

    threadgroup half * As = (threadgroup half *) shmem;         // [BK][BM]
    threadgroup half * Bs = (threadgroup half *) shmem + BK*BM; // [BK][BN]

    const short tc = tiitg % NTX;
    const short tr = tiitg / NTX;

    float4 acc[TM][TN/4];
    FOR_UNROLL (short i = 0; i < TM; i++) {
        FOR_UNROLL (short j = 0; j < TN/4; j++) {
            acc[i][j] = float4(0.0f);
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // stage + dequantize the A (expert weights) tile to half, K-major
        for (short it = tiitg; it < A_ITEMS; it += NTHREAD) {
            const short m    = it % BM;
            const short sb   = it / BM;
            const int   kpos = k0 + sb*16;

            if (r0 + m < M) {
                if (is_same<T0_4x4, block_q>::value) {
                    device const T0 * row_ptr = (device const T0 *)(src0 + args.nb01*(r0 + m) + offset0);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        As[(sb*16 + i)*BM + m] = (kpos + i < K) ? (half) row_ptr[kpos + i] : (half) 0;
                    }
                } else if (kpos < K) {
                    const int   block_idx = kpos / (16*nl);
                    const short il        = (kpos/16) % nl;
                    device const block_q * row_ptr = (device const block_q *)(src0 + args.nb01*(r0 + m) + offset0);
                    T0_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        As[(sb*16 + i)*BM + m] = (kpos + i < K) ? (half) temp_a[i/4][i%4] : (half) 0;
                    }
                } else {
                    // past the end of the row: do not read the next block
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        As[(sb*16 + i)*BM + m] = (half) 0;
                    }
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    As[(sb*16 + i)*BM + m] = (half) 0;
                }
            }
        }

        // stage the B tile: gather one src1 row per token via hids
        for (short nloc = tiitg; nloc < BN; nloc += NTHREAD) {
            if (r1 + nloc < neh1) {
                const int id  = ids_i32[im*args.ne21 + r1 + nloc];
                const int i11 = (id % args.ne20) % args.ne11;
                const int i12 = (id / args.ne20);
                device const T1 * yptr = (device const T1 *)(src1 + args.nb12*i12 + args.nb11*i11) + k0;
                FOR_UNROLL (short kk = 0; kk < BK; kk++) {
                    Bs[kk*BN + nloc] = (k0 + kk < K) ? (half) ((float) yptr[kk] * s1_inv) : (half) 0;
                }
            } else {
                FOR_UNROLL (short kk = 0; kk < BK; kk++) {
                    Bs[kk*BN + nloc] = (half) 0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        half4 hacc[TM][TN/4];
        FOR_UNROLL (short i = 0; i < TM; i++) {
            FOR_UNROLL (short j = 0; j < TN/4; j++) {
                hacc[i][j] = half4(0.0h);
            }
        }

        FOR_UNROLL (short kk = 0; kk < BK; kk++) {
            half4 a_reg[TM/4];
            half4 b_reg[TN/4];

            FOR_UNROLL (short i = 0; i < TM/4; i++) {
                a_reg[i] = *((threadgroup half4 *)(As + kk*BM + tr*TM) + i);
            }
            FOR_UNROLL (short j = 0; j < TN/4; j++) {
                b_reg[j] = *((threadgroup half4 *)(Bs + kk*BN + tc*TN) + j);
            }
            FOR_UNROLL (short i = 0; i < TM; i++) {
                const half a = a_reg[i/4][i%4];
                FOR_UNROLL (short j = 0; j < TN/4; j++) {
                    hacc[i][j] += a*b_reg[j];
                }
            }
        }

        FOR_UNROLL (short i = 0; i < TM; i++) {
            FOR_UNROLL (short j = 0; j < TN/4; j++) {
                acc[i][j] += float4(hacc[i][j]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // scatter the result back to dst via hids (de-route)
    device float * dst_f32 = (device float *) dst;

    FOR_UNROLL (short i = 0; i < TM; i++) {
        const int m = r0 + tr*TM + i;
        if (m < M) {
            FOR_UNROLL (short j = 0; j < TN/4; j++) {
                const float4 rr = acc[i][j];
                float lanes[4] = { rr.x, rr.y, rr.z, rr.w };
                FOR_UNROLL (short lane = 0; lane < 4; lane++) {
                    const int nloc = tc*TN + 4*j + lane;
                    if (r1 + nloc < neh1) {
                        const int id  = ids_i32[im*args.ne21 + r1 + nloc];
                        const int ide = id % args.ne20;
                        const int idt = id / args.ne20;
                        dst_f32[m + ide*args.ne0 + (size_t) idt*args.ne1*args.ne0] = lanes[lane] * s1_scale;
                    }
                }
            }
        }
    }
}

typedef decltype(kernel_mul_mm_id_tiled<half, half4x4, half4x4, 1, dequantize_f16, float>) mul_mm_id_tiled_t;

template [[host_name("kernel_mul_mm_id_tiled_f32_f32")]]   kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, float4x4,   1,     dequantize_f32,   float>;
template [[host_name("kernel_mul_mm_id_tiled_f16_f32")]]   kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, half4x4,    1,     dequantize_f16,   float>;
template [[host_name("kernel_mul_mm_id_tiled_q4_0_f32")]]  kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_q4_0,  2,     dequantize_q4_0,  float>;
template [[host_name("kernel_mul_mm_id_tiled_q4_1_f32")]]  kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_q4_1,  2,     dequantize_q4_1,  float>;
template [[host_name("kernel_mul_mm_id_tiled_q5_0_f32")]]  kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_q5_0,  2,     dequantize_q5_0,  float>;
template [[host_name("kernel_mul_mm_id_tiled_q5_1_f32")]]  kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_q5_1,  2,     dequantize_q5_1,  float>;
template [[host_name("kernel_mul_mm_id_tiled_q8_0_f32")]]  kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_q8_0,  2,     dequantize_q8_0,  float>;
template [[host_name("kernel_mul_mm_id_tiled_q2_K_f32")]]  kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_q2_K,  QK_NL, dequantize_q2_K,  float>;
template [[host_name("kernel_mul_mm_id_tiled_q3_K_f32")]]  kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_q3_K,  QK_NL, dequantize_q3_K,  float>;
template [[host_name("kernel_mul_mm_id_tiled_q4_K_f32")]]  kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_q4_K,  QK_NL, dequantize_q4_K,  float>;
template [[host_name("kernel_mul_mm_id_tiled_q5_K_f32")]]  kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_q5_K,  QK_NL, dequantize_q5_K,  float>;
template [[host_name("kernel_mul_mm_id_tiled_q6_K_f32")]]  kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_q6_K,  QK_NL, dequantize_q6_K,  float>;
template [[host_name("kernel_mul_mm_id_tiled_iq3_s_f32")]] kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_iq3_s, QK_NL, dequantize_iq3_s, float>;
template [[host_name("kernel_mul_mm_id_tiled_mxfp4_f32")]]   kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_mxfp4,   2,     dequantize_mxfp4,   float>;
template [[host_name("kernel_mul_mm_id_tiled_iq2_xxs_f32")]] kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float>;
template [[host_name("kernel_mul_mm_id_tiled_iq2_xs_f32")]]  kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float>;
template [[host_name("kernel_mul_mm_id_tiled_iq2_s_f32")]]   kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_iq2_s,   QK_NL, dequantize_iq2_s,   float>;
template [[host_name("kernel_mul_mm_id_tiled_iq3_xxs_f32")]] kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float>;
template [[host_name("kernel_mul_mm_id_tiled_iq1_s_f32")]]   kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_iq1_s,   QK_NL, dequantize_iq1_s,   float>;
template [[host_name("kernel_mul_mm_id_tiled_iq1_m_f32")]]   kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_iq1_m,   QK_NL, dequantize_iq1_m,   float>;
template [[host_name("kernel_mul_mm_id_tiled_iq4_nl_f32")]]  kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_iq4_nl,  2,     dequantize_iq4_nl,  float>;
template [[host_name("kernel_mul_mm_id_tiled_iq4_xs_f32")]]  kernel mul_mm_id_tiled_t kernel_mul_mm_id_tiled<half, half4x4, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float>;

template<short ne20> // n_expert_used
kernel void kernel_mul_mm_id_map0(
        constant ggml_metal_kargs_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg; // expert id

    uint32_t n_all = 0;

    device int32_t * ids_i32 = (device int32_t *) hids + ide*args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) { // n_tokens
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem + tpitg*ne20;

            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids = (threadgroup const uint16_t *) shmem + t*ne20;

            short sel = 0;
            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sel += (sids[i20] == ide)*(i20 + 1);
            }

            ids_i32[n_all] = (i21 + t)*ne20 + sel - 1;

            n_all += sel > 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) (htpe);
    tpe_u32[ide] = n_all;
}

kernel void kernel_mul_mm_id_amax_part_f32(
        constant ggml_metal_kargs_mul_mm_id_amax & args,
        device   const char * src1,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const int nrow = args.ne01*args.ne02;

    float lmax = 0.0f;

    for (int ir = tgpig; ir < nrow; ir += N_MM_NPART_AMAX) {
        const int i01 = ir % args.ne01;
        const int i02 = ir / args.ne01;

        device const float * row = (device const float *) (src1 + i02*args.nb02 + i01*args.nb01);

        for (int i00 = tiitg; i00 < args.ne00; i00 += ntg) {
            lmax = max(lmax, fabs(row[i00]));
        }
    }

    float amax = simd_max(lmax);

    threadgroup float * shared_amax = (threadgroup float *) shmem;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_amax[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_amax[sgitg] = amax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        amax = shared_amax[tiisg];
        amax = simd_max(amax);
    }

    if (tiitg == 0) {
        ((device float *) (dst + 8))[tgpig] = amax;
    }
}

kernel void kernel_mul_mm_id_amax_f32(
        device char * dst,
        ushort tiitg[[thread_index_in_threadgroup]]) {
    device const float * part = (device const float *) (dst + 8);

    float amax = 0.0f;

    for (int i = tiitg; i < N_MM_NPART_AMAX; i += N_SIMDWIDTH) {
        amax = max(amax, part[i]);
    }

    amax = simd_max(amax);

    if (tiitg == 0) {
        // leave a comfortable margin below the f16 max of 65504
        float scale = 1.0f;

        // isfinite: src1 already inf/nan is not ours to fix - keep the
        // scale at 1.0 instead of turning it into a different failure
        if (isfinite(amax) && amax > 32768.0f) {
            scale = exp2(ceil(log2(amax)) - 15.0f);
        }

        device float * d = (device float *) dst;

        d[0] = 1.0f/scale; // exact: scale is a power of two
        d[1] = scale;
    }
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;

template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_5" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<5>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;
template [[host_name("kernel_mul_mm_id_map0_ne20_22")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<22>;

//
// MoE expert matmul for GPUs without simdgroup-mm or tensor API (GGML_METAL_MOE_WL)
//

// kernel_mul_mm_id_map0 plus the work list of kernel_mul_mm_id_wl. one threadgroup, one thread per expert
//   htpe[e]          = number of tokens routed to expert e
//   hids[e*ne21 + i] = token*ne20 + slot of the i-th of them, in token order (only i < htpe[e] is written)
//   hwl[0]           = number of work items
//   hwl[1 + i]       = (e << 16) | j: tokens N_MM_ID_WL_BN*j .. N_MM_ID_WL_BN*j + N_MM_ID_WL_BN - 1 of expert e, expert-major
template<short ne20> // n_expert_used
kernel void kernel_mul_mm_id_map_wl(
        constant ggml_metal_kargs_mul_mm_id_map0 & args,
        device const char * src2,
        device       char * htpe,
        device       char * hids,
        device       char * hwl,
        threadgroup  char * shmem [[threadgroup(0)]],
        ushort tpitg [[thread_position_in_threadgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    const int ide = tpitg; // expert id
    const int nth = ntg;   // n_expert

    threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem; // [nth][ne20] expert ids of nth tokens
    threadgroup uint32_t * scan = (threadgroup uint32_t *) shmem; // [nth], after the token loop

    device int32_t * ids_i32 = (device int32_t *) hids + (uint64_t) ide*args.ne21;

    int n_all = 0;

    for (int i21 = 0; i21 < args.ne21; i21 += nth) {
        if (i21 + ide < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (uint64_t) (i21 + ide)*args.nb21);

            FOR_UNROLL (int i20 = 0; i20 < ne20; i20++) {
                sids[ide*ne20 + i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int nt = min(nth, args.ne21 - i21);

        for (int t = 0; t < nt; t++) {
            // the expert ids of a token are expected to be distinct (as for map0); a repeated id
            // takes its first slot only, so no dst row is written twice
            int sel = 0;
            FOR_UNROLL (int i20 = 0; i20 < ne20; i20++) {
                sel = (sel == 0 && sids[t*ne20 + i20] == ide) ? i20 + 1 : sel;
            }

            // store only the hits (map0 stores every token)
            if (sel > 0) {
                ids_i32[n_all] = (i21 + t)*ne20 + sel - 1;
                n_all++;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    ((device uint32_t *) htpe)[ide] = n_all;

    // inclusive scan of the work item counts over the experts
    const uint32_t nwi = (n_all + N_MM_ID_WL_BN - 1)/N_MM_ID_WL_BN;

    scan[ide] = nwi;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int off = 1; off < nth; off *= 2) {
        const uint32_t v = ide >= off ? scan[ide - off] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan[ide] += v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * wl = (device uint32_t *) hwl;

    const uint32_t base = scan[ide] - nwi;

    for (uint32_t j = 0; j < nwi; j++) {
        wl[1 + base + j] = ((uint32_t) ide << 16) | j;
    }

    if (ide == nth - 1) {
        wl[0] = scan[ide];
    }
}

typedef decltype(kernel_mul_mm_id_map_wl<1>) kernel_mul_mm_id_map_wl_t;

template [[host_name("kernel_mul_mm_id_map_wl_ne20_1" )]] kernel kernel_mul_mm_id_map_wl_t kernel_mul_mm_id_map_wl<1>;
template [[host_name("kernel_mul_mm_id_map_wl_ne20_2" )]] kernel kernel_mul_mm_id_map_wl_t kernel_mul_mm_id_map_wl<2>;
template [[host_name("kernel_mul_mm_id_map_wl_ne20_4" )]] kernel kernel_mul_mm_id_map_wl_t kernel_mul_mm_id_map_wl<4>;
template [[host_name("kernel_mul_mm_id_map_wl_ne20_5" )]] kernel kernel_mul_mm_id_map_wl_t kernel_mul_mm_id_map_wl<5>;
template [[host_name("kernel_mul_mm_id_map_wl_ne20_6" )]] kernel kernel_mul_mm_id_map_wl_t kernel_mul_mm_id_map_wl<6>;
template [[host_name("kernel_mul_mm_id_map_wl_ne20_8" )]] kernel kernel_mul_mm_id_map_wl_t kernel_mul_mm_id_map_wl<8>;
template [[host_name("kernel_mul_mm_id_map_wl_ne20_10")]] kernel kernel_mul_mm_id_map_wl_t kernel_mul_mm_id_map_wl<10>;
template [[host_name("kernel_mul_mm_id_map_wl_ne20_16")]] kernel kernel_mul_mm_id_map_wl_t kernel_mul_mm_id_map_wl<16>;
template [[host_name("kernel_mul_mm_id_map_wl_ne20_22")]] kernel kernel_mul_mm_id_map_wl_t kernel_mul_mm_id_map_wl<22>;

// grouped expert matmul over the work list of kernel_mul_mm_id_map_wl (GGML_METAL_MOE_WL)
// threadgroup (x, y) = (128 rows, work item = expert e and up to 32 of its tokens), 128 threads = 4 simdgroups
// each weight is dequantized once per work item. K-major packed half2 words, as kernel_mul_mm_wc:
//   As[k][w ^ g*RPW]         = (A[w][k], A[w + 64][k]), g = k/16 (swizzle for this staging map)
//   Bs[kp][n ^ ((kp/4)%4)*8] = (B[n][2kp], B[n][2kp + 1])
// simdgroup w owns tokens 8w..8w+7 for all 128 rows: it gathers and multiplies them only if the item has any
template<
    typename T0, typename T0_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread T0_4x4 &),
    int BK>
kernel void kernel_mul_mm_id_wl(
        constant ggml_metal_kargs_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device const char * hwl,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig   [[threadgroup_position_in_grid]],
        ushort tiitg_u [[thread_index_in_threadgroup]]) {
    constexpr int BM  = N_MM_ID_WL_BM;
    constexpr int BN  = N_MM_ID_WL_BN;
    constexpr int NW  = BM/2;  // As words per k
    constexpr int G   = BK/16; // k groups per chunk
    constexpr int RPW = 32/G;  // row pairs per simdgroup per A staging pass
    constexpr int NPA = G/2;   // A staging passes
    constexpr int NPB = BK/32; // B staging passes

    static_assert(BM == 128 && BN == 32 && (BK == 32 || BK == 64), "unsupported tile");

    // all index math in int: short address math miscompiles on this backend
    const int tiitg = tiitg_u;

    device const uint32_t * wl = (device const uint32_t *) hwl;

    // the grid is sized for the most work items possible
    const int it = tgpig.y;
    if (it >= (int) wl[0]) {
        return;
    }

    const uint32_t item = wl[1 + it];

    const int e  = item >> 16;
    const int t0 = (int) (item & 0xFFFF)*BN;
    const int nt = min(BN, (int) ((device const uint32_t *) htpe)[e] - t0);

    const int K  = args.ne00;
    const int M  = args.ne0;
    const int ra = tgpig.x*BM;

    const uint64_t offset0 = (uint64_t) e*args.nb02;

    device const int32_t * ids = (device const int32_t *) hids + (uint64_t) e*args.ne21 + t0;

    threadgroup uint * As = (threadgroup uint *) shmem;         // [BK][NW]
    threadgroup uint * Bs = (threadgroup uint *) shmem + BK*NW; // [BK/2][BN]

    const int w    = tiitg/32;
    const int lane = tiitg%32;

    // compute: rows 4rg..4rg+3 and 4rg+64..4rg+67, tokens n0..n0+3
    const int  rg     = lane%16;
    const int  n0     = 8*w + 4*(lane/16);
    const bool active = 8*w < nt; // uniform in the simdgroup

    // A staging: k group ag of rows rp and rp + 64
    const int ag = lane%G;

    // B staging: token bn, k octets bo + 4p
    const int bn = 8*w + lane/4;
    const int bo = lane%4;

    device const float * y = (device const float *) src1;
    if (bn < nt) {
        const int id  = ids[bn];
        const int i11 = (id % args.ne20) % args.ne11;
        const int i12 =  id / args.ne20;

        y = (device const float *) (src1 + (uint64_t) i12*args.nb12 + (uint64_t) i11*args.nb11);
    }

    float4 acc[4][2]; // [token n0 + j][rows {4rg, 4rg+64, 4rg+1, 4rg+65} | {4rg+2, 4rg+66, 4rg+3, 4rg+67}]
    FOR_UNROLL (int j = 0; j < 4; j++) {
        acc[j][0] = float4(0.0f);
        acc[j][1] = float4(0.0f);
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        uint wa[NPA][16];
        FOR_UNROLL (int p = 0; p < NPA; p++) {
            const int rp   = RPW*(4*p + w) + lane/G;
            const int kpos = k0 + 16*ag;

            const T0_4x4 r0 = mm_wc_load_a<T0, T0_4x4, block_q, nl, dequantize_func>(src0, args.nb01, offset0, ra + rp,      M, K, kpos);
            const T0_4x4 r1 = mm_wc_load_a<T0, T0_4x4, block_q, nl, dequantize_func>(src0, args.nb01, offset0, ra + rp + 64, M, K, kpos);

            FOR_UNROLL (int i = 0; i < 16; i++) {
                wa[p][i] = as_type<uint>(half2(r0[i/4][i%4], r1[i/4][i%4]));
            }
        }

        // zero for tokens past the item
        uint wb[NPB][4];
        FOR_UNROLL (int p = 0; p < NPB; p++) {
            float4 x0 = float4(0.0f);
            float4 x1 = float4(0.0f);
            if (bn < nt) {
                device const float4 * py = (device const float4 *) (y + k0 + 8*(bo + 4*p));
                x0 = py[0];
                x1 = py[1];
            }
            wb[p][0] = as_type<uint>(half2(x0.xy));
            wb[p][1] = as_type<uint>(half2(x0.zw));
            wb[p][2] = as_type<uint>(half2(x1.xy));
            wb[p][3] = as_type<uint>(half2(x1.zw));
        }

        // early staging: the loads above overlap the compute of the slower simdgroups
        if (k0 > 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // every store of a simdgroup hits 32 distinct banks
        FOR_UNROLL (int p = 0; p < NPA; p++) {
            const int rp = RPW*(4*p + w) + lane/G;
            threadgroup uint * pa = As + 16*ag*NW + (rp ^ (ag*RPW));
            FOR_UNROLL (int i = 0; i < 16; i++) {
                pa[i*NW] = wa[p][i];
            }
        }
        // the B words of a simdgroup are read only by that simdgroup
        if (active) {
            FOR_UNROLL (int p = 0; p < NPB; p++) {
                threadgroup uint * pb = Bs + 4*(bo + 4*p)*BN + (bn ^ (bo << 3));
                FOR_UNROLL (int c = 0; c < 4; c++) {
                    pb[c*BN] = wb[p][c];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            half4 hacc[4][2];
            FOR_UNROLL (int j = 0; j < 4; j++) {
                hacc[j][0] = half4(0.0h);
                hacc[j][1] = half4(0.0h);
            }

            FOR_UNROLL (int kp = 0; kp < BK/2; kp++) {
                const int xa = (kp/8)*RPW;     // k group of k = 2kp, 2kp + 1
                const int xb = ((kp/4) % 4)*8; // k octet of k pair kp

                const uint4 ua0 = *((threadgroup const uint4 *) (As + (2*kp + 0)*NW + ((4*rg) ^ xa)));
                const uint4 ua1 = *((threadgroup const uint4 *) (As + (2*kp + 1)*NW + ((4*rg) ^ xa)));
                const uint4 ub  = *((threadgroup const uint4 *) (Bs + kp*BN + (n0 ^ xb)));

                const half4 a00 = as_type<half4>(ua0.xy); // k = 2kp:     rows {4rg, 4rg+64, 4rg+1, 4rg+65}
                const half4 a01 = as_type<half4>(ua0.zw); // k = 2kp:     rows {4rg+2, 4rg+66, 4rg+3, 4rg+67}
                const half4 a10 = as_type<half4>(ua1.xy); // k = 2kp + 1
                const half4 a11 = as_type<half4>(ua1.zw);

                FOR_UNROLL (int j = 0; j < 4; j++) {
                    const half2 b = as_type<half2>(ub[j]); // token n0 + j at k = 2kp, 2kp + 1
                    hacc[j][0] = fma(half4(b.x), a00, hacc[j][0]);
                    hacc[j][1] = fma(half4(b.x), a01, hacc[j][1]);
                    hacc[j][0] = fma(half4(b.y), a10, hacc[j][0]);
                    hacc[j][1] = fma(half4(b.y), a11, hacc[j][1]);
                }
            }

            // fold the BK half products into FP32, as kernel_mul_mm_wc
            FOR_UNROLL (int j = 0; j < 4; j++) {
                acc[j][0] += float4(hacc[j][0]);
                acc[j][1] += float4(hacc[j][1]);
            }
        }
    }

    if (!active) {
        return;
    }

    device float * dst_f32 = (device float *) dst;

    const bool m_vec = M % 4 == 0;

    // scatter to dst via hids, as kernel_mul_mm_id_tiled
    FOR_UNROLL (int j = 0; j < 4; j++) {
        const int n = n0 + j;
        if (n < nt) {
            const int id  = ids[n];
            const int ide = id % args.ne20;
            const int idt = id / args.ne20;

            device float * D = dst_f32 + (uint64_t) idt*args.ne1*args.ne0 + (uint64_t) ide*args.ne0;

            const float4 lo = float4(acc[j][0].x, acc[j][0].z, acc[j][1].x, acc[j][1].z); // rows 4rg..4rg+3
            const float4 hi = float4(acc[j][0].y, acc[j][0].w, acc[j][1].y, acc[j][1].w); // rows 4rg+64..4rg+67

            FOR_UNROLL (int h = 0; h < 2; h++) {
                const int    m = ra + 4*rg + 64*h;
                const float4 v = h == 0 ? lo : hi;
                if (m_vec && m < M) {
                    *((device float4 *) (D + m)) = v;
                } else {
                    FOR_UNROLL (int c = 0; c < 4; c++) {
                        if (m + c < M) {
                            D[m + c] = v[c];
                        }
                    }
                }
            }
        }
    }
}

typedef decltype(kernel_mul_mm_id_wl<half, half4x4, block_q4_0, 2, dequantize_q4_0, 32>) mul_mm_id_wl_t;

template [[host_name("kernel_mul_mm_id_wl_q4_0_f32_bk32")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q4_0,    2,     dequantize_q4_0,    32>;
template [[host_name("kernel_mul_mm_id_wl_q4_1_f32_bk32")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q4_1,    2,     dequantize_q4_1,    32>;
template [[host_name("kernel_mul_mm_id_wl_q5_0_f32_bk32")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q5_0,    2,     dequantize_q5_0,    32>;
template [[host_name("kernel_mul_mm_id_wl_q5_1_f32_bk32")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q5_1,    2,     dequantize_q5_1,    32>;
template [[host_name("kernel_mul_mm_id_wl_q8_0_f32_bk32")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q8_0,    2,     dequantize_q8_0,    32>;
template [[host_name("kernel_mul_mm_id_wl_mxfp4_f32_bk32")]]   kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_mxfp4,   2,     dequantize_mxfp4,   32>;
template [[host_name("kernel_mul_mm_id_wl_q2_K_f32_bk32")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q2_K,    QK_NL, dequantize_q2_K,    32>;
template [[host_name("kernel_mul_mm_id_wl_q3_K_f32_bk32")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q3_K,    QK_NL, dequantize_q3_K,    32>;
template [[host_name("kernel_mul_mm_id_wl_q4_K_f32_bk32")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q4_K,    QK_NL, dequantize_q4_K,    32>;
template [[host_name("kernel_mul_mm_id_wl_q5_K_f32_bk32")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q5_K,    QK_NL, dequantize_q5_K,    32>;
template [[host_name("kernel_mul_mm_id_wl_q6_K_f32_bk32")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q6_K,    QK_NL, dequantize_q6_K,    32>;
template [[host_name("kernel_mul_mm_id_wl_iq2_xxs_f32_bk32")]] kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, 32>;
template [[host_name("kernel_mul_mm_id_wl_iq2_xs_f32_bk32")]]  kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  32>;
template [[host_name("kernel_mul_mm_id_wl_iq2_s_f32_bk32")]]   kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq2_s,   QK_NL, dequantize_iq2_s,   32>;
template [[host_name("kernel_mul_mm_id_wl_iq3_xxs_f32_bk32")]] kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, 32>;
template [[host_name("kernel_mul_mm_id_wl_iq3_s_f32_bk32")]]   kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq3_s,   QK_NL, dequantize_iq3_s,   32>;
template [[host_name("kernel_mul_mm_id_wl_iq1_s_f32_bk32")]]   kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq1_s,   QK_NL, dequantize_iq1_s,   32>;
template [[host_name("kernel_mul_mm_id_wl_iq1_m_f32_bk32")]]   kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq1_m,   QK_NL, dequantize_iq1_m,   32>;
template [[host_name("kernel_mul_mm_id_wl_iq4_nl_f32_bk32")]]  kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq4_nl,  2,     dequantize_iq4_nl,  32>;
template [[host_name("kernel_mul_mm_id_wl_iq4_xs_f32_bk32")]]  kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  32>;

template [[host_name("kernel_mul_mm_id_wl_q4_0_f32_bk64")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q4_0,    2,     dequantize_q4_0,    64>;
template [[host_name("kernel_mul_mm_id_wl_q4_1_f32_bk64")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q4_1,    2,     dequantize_q4_1,    64>;
template [[host_name("kernel_mul_mm_id_wl_q5_0_f32_bk64")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q5_0,    2,     dequantize_q5_0,    64>;
template [[host_name("kernel_mul_mm_id_wl_q5_1_f32_bk64")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q5_1,    2,     dequantize_q5_1,    64>;
template [[host_name("kernel_mul_mm_id_wl_q8_0_f32_bk64")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q8_0,    2,     dequantize_q8_0,    64>;
template [[host_name("kernel_mul_mm_id_wl_mxfp4_f32_bk64")]]   kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_mxfp4,   2,     dequantize_mxfp4,   64>;
template [[host_name("kernel_mul_mm_id_wl_q2_K_f32_bk64")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q2_K,    QK_NL, dequantize_q2_K,    64>;
template [[host_name("kernel_mul_mm_id_wl_q3_K_f32_bk64")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q3_K,    QK_NL, dequantize_q3_K,    64>;
template [[host_name("kernel_mul_mm_id_wl_q4_K_f32_bk64")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q4_K,    QK_NL, dequantize_q4_K,    64>;
template [[host_name("kernel_mul_mm_id_wl_q5_K_f32_bk64")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q5_K,    QK_NL, dequantize_q5_K,    64>;
template [[host_name("kernel_mul_mm_id_wl_q6_K_f32_bk64")]]    kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_q6_K,    QK_NL, dequantize_q6_K,    64>;
template [[host_name("kernel_mul_mm_id_wl_iq2_xxs_f32_bk64")]] kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, 64>;
template [[host_name("kernel_mul_mm_id_wl_iq2_xs_f32_bk64")]]  kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  64>;
template [[host_name("kernel_mul_mm_id_wl_iq2_s_f32_bk64")]]   kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq2_s,   QK_NL, dequantize_iq2_s,   64>;
template [[host_name("kernel_mul_mm_id_wl_iq3_xxs_f32_bk64")]] kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, 64>;
template [[host_name("kernel_mul_mm_id_wl_iq3_s_f32_bk64")]]   kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq3_s,   QK_NL, dequantize_iq3_s,   64>;
template [[host_name("kernel_mul_mm_id_wl_iq1_s_f32_bk64")]]   kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq1_s,   QK_NL, dequantize_iq1_s,   64>;
template [[host_name("kernel_mul_mm_id_wl_iq1_m_f32_bk64")]]   kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq1_m,   QK_NL, dequantize_iq1_m,   64>;
template [[host_name("kernel_mul_mm_id_wl_iq4_nl_f32_bk64")]]  kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq4_nl,  2,     dequantize_iq4_nl,  64>;
template [[host_name("kernel_mul_mm_id_wl_iq4_xs_f32_bk64")]]  kernel mul_mm_id_wl_t kernel_mul_mm_id_wl<half, half4x4, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  64>;

template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm_id(
        constant ggml_metal_kargs_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        device const char * amax,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

#ifdef GGML_METAL_HAS_TENSOR
    threadgroup float * sc = (threadgroup float *)(shmem);
#endif

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z; // expert
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    // skip the upper half of the token tile when the expert did not fill it
    constexpr short NR1H = NR1/2;

    const bool has_hi = nr1 > NR1H;

    // power-of-two rescaling
    const float s1_inv   = FC_mul_mm_id_amax ? ((device const float *) amax)[0] : 1.0f;
    const float s1_scale = FC_mul_mm_id_amax ? ((device const float *) amax)[1] : 1.0f;

#ifndef GGML_METAL_HAS_TENSOR
    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // simdgroups 2,3 own rows NR1H..NR1-1
    const bool sg_active = has_hi || sgitg < 2;
#else
    auto tA  = tensor<threadgroup S0, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));

    // sb is [NR1][NK] row-major
    auto tB0 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb,             dextents<int32_t, 2>(NK, NR1H));
    auto tB1 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb + NR1H*NK,   dextents<int32_t, 2>(NK, NR1H));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1H, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT0 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB0), float>();
    auto cT1 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB1), float>();
#endif

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
#ifndef GGML_METAL_HAS_TENSOR
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : (S0) 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) (*((device T1 *) y + i) * (T1) s1_inv) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)((*((device T1_2x4 *) y)) * (T1) s1_inv);
        }
#else
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
                //const short lx = (tiitg/NL1)%8;
                //const short ly = i;

                *(sb + NK*(8*sy + ly) + 8*sx + lx) = loop_k + iy + i < args.ne00 ? (S1) (*((device T1 *) y + i) * (T1) s1_inv) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            //const short lx = i;
            const short ly = (tiitg/NL1)%8;
            //const short lx = (tiitg/NL1)%8;
            //const short ly = i;

            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = (S1_2x4)((*((device T1_2x4 *) y)) * (T1) s1_inv);
        }
#endif

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifndef GGML_METAL_HAS_TENSOR
        if (sg_active) {
            // load matrices from threadgroup memory and conduct outer products
            threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
            threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

            FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 4; i++) {
                    simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 2; i++) {
                    simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 8; i++){
                    simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
                }

                lsma += 8*64;
                lsmb += 4*64;
            }
        }
#else
        auto sA  = tA.slice(0, 0);
        auto sB0 = tB0.slice(0, 0);

        mm.run(sB0, sA, cT0);

        if (has_hi) {
            auto sB1 = tB1.slice(0, 0);

            mm.run(sB1, sA, cT1);
        }
#endif
    }

    // block is smaller than 64x32, we should avoid writing data outside of the matrix
    threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef GGML_METAL_HAS_TENSOR
    auto tC0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc,             dextents<int32_t, 2>(NR0, NR1H));
    cT0.store(tC0);

    if (has_hi) {
        auto tC1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc + NR1H*NR0, dextents<int32_t, 2>(NR0, NR1H));
        cT1.store(tC1);
    }
#else
    if (sg_active) {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
    }
#endif

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[im*args.ne21 + r1 + j];

        const short ide = id % args.ne20;
        const short idt = id / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i) * s1_scale;
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i) * s1_scale;
        }
    }
}

//
// matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_q1_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;

template [[host_name("kernel_mul_mm_f32_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_f16_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_q1_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q8_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q3_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q6_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;

//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_id;

template [[host_name("kernel_mul_mm_id_f32_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_f16_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_bf16_f32")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_id_q1_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
#else
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;

template [[host_name("kernel_mul_mm_id_f32_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_f16_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_id_q1_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
#else
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
#endif
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;
