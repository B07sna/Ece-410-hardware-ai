// gemm_tiled.cu — Tiled N×N FP32 GEMM using shared memory (TILE_SIZE = 8)
// Compile: nvcc -O2 -o gemm_tiled gemm_tiled.cu
// Run:     gemm_tiled.exe

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static const int N          = 1024;   // matrix dimension (must be multiple of TILE_SIZE)
static const int TILE_SIZE  = 8;      // shared-memory tile edge
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
// Tiled GEMM kernel
//
// Each thread block owns one TILE_SIZE×TILE_SIZE output tile of C.
// The k-dimension is swept in TILE_SIZE-wide strips:
//
//   for each k-strip t  (t = 0 .. N/TILE_SIZE - 1):
//     1. Cooperatively load A[row_block][t] tile  → shared As[TILE_SIZE][TILE_SIZE]
//     2. Cooperatively load B[t][col_block] tile  → shared Bs[TILE_SIZE][TILE_SIZE]
//     3. __syncthreads()
//     4. Each thread accumulates TILE_SIZE MACs from As and Bs (register only)
//     5. __syncthreads()   (guard before overwriting shared memory)
//
// DRAM loads per element: 1 time each (tile is loaded once and reused TILE_SIZE
// times), giving Traffic = 2N² × 4 bytes versus Naive's 2N³ × 4 bytes.
// Reduction factor = N = 1024.
// ---------------------------------------------------------------------------
__global__ void gemm_tiled(const float * __restrict__ A,
                            const float * __restrict__ B,
                            float       * __restrict__ C,
                            int n)
{
    // Shared-memory tiles — one per matrix operand
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Global row and column this thread is responsible for
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    // Sweep k-dimension in tiles of width TILE_SIZE
    const int num_tiles = n / TILE_SIZE;   // N=1024, TILE_SIZE=8 → 128 tiles
    for (int t = 0; t < num_tiles; ++t) {

        // --- Cooperative tile load ---
        // Each thread loads one element of As and one of Bs.
        // threadIdx.y selects the row within the tile,
        // threadIdx.x selects the column.
        As[threadIdx.y][threadIdx.x] = A[row * n + (t * TILE_SIZE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];

        // Ensure the entire tile is resident before any thread reads it
        __syncthreads();

        // --- Tile dot-product (TILE_SIZE MACs, fully register-resident) ---
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Guard: don't overwrite shared memory before every thread has consumed it
        __syncthreads();
    }

    // Write result (bounds guard not needed when N is a multiple of TILE_SIZE,
    // but included for generality)
    if (row < n && col < n) {
        C[row * n + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Host utilities
// ---------------------------------------------------------------------------
static void init_matrix(float *M, int n, float seed_offset)
{
    for (int i = 0; i < n * n; ++i) {
        M[i] = (float)(rand()) / (float)(RAND_MAX) - 0.5f + seed_offset * 1e-4f;
    }
}

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
    printf("=== Tiled GEMM  (N=%d, tile=%dx%d, FP32) ===\n\n",
           N, TILE_SIZE, TILE_SIZE);

    // ---- Static assertions ----
    static_assert(N % TILE_SIZE == 0,
                  "N must be a multiple of TILE_SIZE for this kernel");

    // ---- Print device info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("    SM count  : %d\n", prop.multiProcessorCount);
    printf("    Global mem: %.1f GB\n",
           (double)prop.totalGlobalMem / (1 << 30));
    printf("    Shared mem: %zu KB per block\n",
           prop.sharedMemPerBlock / 1024);
    printf("    Mem BW    : (see nvidia-smi for bandwidth)\n\n");

    // Shared memory used by this kernel
    size_t smem_used = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    printf("Shared memory used per block: %zu bytes  (2 × %d×%d × 4 B)\n\n",
           smem_used, TILE_SIZE, TILE_SIZE);

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
    // One thread block → one TILE_SIZE×TILE_SIZE output tile
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(N / TILE_SIZE, N / TILE_SIZE);    // 128 × 128 = 16,384 blocks
    printf("Grid : (%d, %d)  Block: (%d, %d)\n",
           grid.x, grid.y, block.x, block.y);
    printf("Total threads : %d\n",
           grid.x * grid.y * block.x * block.y);
    printf("Total blocks  : %d\n\n",
           grid.x * grid.y);

    // ---- CUDA events for timing ----
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // ---- Warm-up ----
    gemm_tiled<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Warm-up run complete.\n\n");

    // ---- Timed runs ----
    float total_ms = 0.0f;
    for (int run = 0; run < NUM_RUNS; ++run) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        gemm_tiled<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaGetLastError());

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_ms += ms;
        printf("  Run %d: %.3f ms\n", run + 1, ms);
    }

    // ---- Results ----
    float  avg_ms  = total_ms / NUM_RUNS;
    double flops   = 2.0 * (double)N * N * N;          // 2N³ FMAs
    double gflops  = flops / (avg_ms * 1.0e6);

    // Tiled DRAM traffic: each element of A and B loaded exactly once
    //   Traffic = 2 × N² × sizeof(float) = 2N² × 4 bytes
    double dram_bytes   = 2.0 * (double)N * N * sizeof(float);
    double bandwidth_gb = dram_bytes / (avg_ms * 1.0e6);

    // Arithmetic intensity from DRAM perspective
    double ai = flops / dram_bytes;   // = N/4 = 256 FLOP/byte for N=1024

    printf("\n--- Tiled GEMM Results ---\n");
    printf("  Matrix size      : %d x %d  (FP32)\n",  N, N);
    printf("  Tile size        : %d x %d\n",           TILE_SIZE, TILE_SIZE);
    printf("  Total FLOPs      : %.3e\n",              flops);
    printf("  DRAM traffic     : %.3f MB  (2N² × 4 B, each element once)\n",
           dram_bytes / (1 << 20));
    printf("  Traffic vs naive : %.0fx less  (N = %d)\n",
           (2.0 * N * N * N * sizeof(float)) / dram_bytes, N);
    printf("  Avg time         : %.3f ms  (%d runs)\n", avg_ms, NUM_RUNS);
    printf("  Performance      : %.2f GFLOPS\n",  gflops);
    printf("  Effective BW     : %.2f GB/s\n",    bandwidth_gb);
    printf("  Arith. intensity : %.1f FLOP/byte  (N/4 = %d/4)\n", ai, N);
    printf("  Bound            : COMPUTE-BOUND\n");
    printf("    (AI >> SRAM ridge; DRAM traffic hidden by pipelining)\n");

    // ---- Spot-check ----
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
