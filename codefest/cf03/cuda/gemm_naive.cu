// gemm_naive.cu — Naive N×N FP32 GEMM: one thread per output element
// Compile: nvcc -O2 -o gemm_naive gemm_naive.cu
// Run:     gemm_naive.exe

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static const int N          = 1024;   // matrix dimension
static const int BLOCK_DIM  = 16;     // threads per block edge (16×16 = 256)
static const int NUM_RUNS   = 5;      // timed iterations (first is warm-up)

// ---------------------------------------------------------------------------
// Error-checking helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Naive kernel: each thread computes one element of C
//
// C[row][col] = sum_k  A[row][k] * B[k][col]
//
// A and B are stored row-major.  Threads in a warp step along consecutive
// columns (col), so C and A reads are coalesced; B reads are strided (column
// access in row-major), which stresses DRAM bandwidth — the classic
// memory-bound pattern the CMAN analysis predicts.
// ---------------------------------------------------------------------------
__global__ void gemm_naive(const float * __restrict__ A,
                            const float * __restrict__ B,
                            float       * __restrict__ C,
                            int n)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;

    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
        acc += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = acc;
}

// ---------------------------------------------------------------------------
// Host utilities
// ---------------------------------------------------------------------------
static void init_matrix(float *M, int n, float seed_offset)
{
    for (int i = 0; i < n * n; ++i) {
        // Values in [-0.5, 0.5] scaled by seed_offset to break symmetry
        M[i] = (float)(rand()) / (float)(RAND_MAX) - 0.5f + seed_offset * 1e-4f;
    }
}

// Spot-check a handful of C elements against CPU reference to catch bugs.
// Only practical for small N; for N=1024 we check SPOT_CHECKS random cells.
static void spot_check(const float *A, const float *B, const float *C, int n,
                        int checks)
{
    srand(42);
    int errors = 0;
    for (int c = 0; c < checks; ++c) {
        int row = rand() % n;
        int col = rand() % n;
        float ref = 0.0f;
        for (int k = 0; k < n; ++k) {
            ref += A[row * n + k] * B[k * n + col];
        }
        float rel = fabsf(ref - C[row * n + col]) / (fabsf(ref) + 1e-6f);
        if (rel > 1e-3f) {
            fprintf(stderr, "  MISMATCH at [%d][%d]: ref=%.6f got=%.6f (rel=%.2e)\n",
                    row, col, ref, C[row * n + col], rel);
            ++errors;
        }
    }
    if (errors == 0)
        printf("  Spot-check (%d cells): PASS\n", checks);
    else
        printf("  Spot-check (%d cells): %d FAILURES\n", checks, errors);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(void)
{
    printf("=== Naive GEMM  (N=%d, block=%dx%d, FP32) ===\n\n",
           N, BLOCK_DIM, BLOCK_DIM);

    // ---- Print device info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("    SM count  : %d\n", prop.multiProcessorCount);
    printf("    Global mem: %.1f GB\n",
           (double)prop.totalGlobalMem / (1 << 30));
    printf("    Mem BW    : (see nvidia-smi for bandwidth)\n\n");

    // ---- Allocate host memory ----
    const size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float *)malloc(bytes);
    float *hB = (float *)malloc(bytes);
    float *hC = (float *)malloc(bytes);
    if (!hA || !hB || !hC) { fprintf(stderr, "malloc failed\n"); return 1; }

    srand(1234);
    init_matrix(hA, N, 1.0f);
    init_matrix(hB, N, 2.0f);

    // ---- Allocate device memory ----
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // ---- Grid / block dimensions ----
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM,
              (N + BLOCK_DIM - 1) / BLOCK_DIM);
    printf("Grid : (%d, %d)  Block: (%d, %d)\n",
           grid.x, grid.y, block.x, block.y);
    printf("Total threads: %d\n\n",
           grid.x * grid.y * block.x * block.y);

    // ---- CUDA events for timing ----
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // ---- Warm-up (run #0, not timed in the average) ----
    gemm_naive<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Warm-up run complete.\n\n");

    // ---- Timed runs ----
    float total_ms = 0.0f;
    for (int run = 0; run < NUM_RUNS; ++run) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        gemm_naive<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaGetLastError());

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_ms += ms;
        printf("  Run %d: %.3f ms\n", run + 1, ms);
    }

    // ---- Results ----
    float avg_ms   = total_ms / NUM_RUNS;
    double flops   = 2.0 * (double)N * N * N;          // 2N³ FMAs
    double gflops  = flops / (avg_ms * 1.0e6);         // GFLOP/s

    // DRAM traffic: each A element loaded N times, each B element loaded N times
    //   Traffic = 2 * N^2 * N * sizeof(float) = 2N³ * 4 bytes
    double dram_bytes   = 2.0 * (double)N * N * N * sizeof(float);
    double bandwidth_gb = dram_bytes / (avg_ms * 1.0e6);  // GB/s achieved

    printf("\n--- Naive GEMM Results ---\n");
    printf("  Matrix size      : %d x %d  (FP32)\n",  N, N);
    printf("  Total FLOPs      : %.3e\n",  flops);
    printf("  DRAM traffic     : %.3f MB  (2N³ × 4 B, no reuse)\n",
           dram_bytes / (1 << 20));
    printf("  Avg time         : %.3f ms  (%d runs)\n", avg_ms, NUM_RUNS);
    printf("  Performance      : %.2f GFLOPS\n",  gflops);
    printf("  Effective BW     : %.2f GB/s\n",    bandwidth_gb);
    printf("  Bound            : MEMORY-BOUND\n");
    printf("    (AI = %.3f FLOP/byte — far below ridge point)\n",
           flops / dram_bytes);

    // ---- Copy result back and spot-check ----
    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    printf("\nSpot-checking against CPU reference (8 cells):\n");
    spot_check(hA, hB, hC, N, 8);

    // ---- Cleanup ----
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);

    return 0;
}
