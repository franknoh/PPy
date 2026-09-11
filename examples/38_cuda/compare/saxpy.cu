// saxpy over sixteen million doubles and a per-block max: CUDA C, timed with events.
#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 24)

__device__ double fma3(double a, double x, double y) { return a * x + y; }

__global__ void saxpy(int n, double a, const double *x, double *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fma3(a, x[i], y[i]);
}

__global__ void block_max(const double *x, double *out) {
    __shared__ double parked[64];
    int tid = threadIdx.x;
    parked[tid] = x[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();
    double mine = parked[tid];
    double other = __shfl_xor_sync(0xffffffffu, mine, 1);
    mine = other > mine ? other : mine;
    parked[tid] = mine;
    __syncthreads();
    if (tid == 0) {
        double best = parked[0];
        for (int k = 1; k < blockDim.x; k++) best = parked[k] > best ? parked[k] : best;
        out[blockIdx.x] = best;
    }
}

static float timed(void (*run)(), const char *label) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float best = 1e9f;
    run();
    cudaDeviceSynchronize();
    for (int r = 0; r < 10; r++) {
        cudaEventRecord(start);
        run();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best) best = ms;
    }
    printf("# %s: %.3f ms\n", label, best);
    return best;
}

static double *dx, *dy, *dv, *dout, *hx, *hy, *hv, *hout;
static void run_saxpy() { saxpy<<<(N + 255) / 256, 256>>>(N, 2.0, dx, dy); }
static void run_block_max() { block_max<<<N / 64, 64>>>(dv, dout); }

/* The same launches over host arrays: the inputs copied in, the results copied out. */
static void run_saxpy_copying() {
    cudaMemcpy(dx, hx, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, N * sizeof(double), cudaMemcpyHostToDevice);
    run_saxpy();
    cudaMemcpy(hy, dy, N * sizeof(double), cudaMemcpyDeviceToHost);
}

static void run_block_max_copying() {
    cudaMemcpy(dv, hv, N * sizeof(double), cudaMemcpyHostToDevice);
    run_block_max();
    cudaMemcpy(hout, dout, N / 64 * sizeof(double), cudaMemcpyDeviceToHost);
}

int main() {
    hx = new double[N]; hy = new double[N]; hv = new double[N]; hout = new double[N / 64];
    for (int i = 0; i < N; i++) { hx[i] = (double)i; hy[i] = 1.0; hv[i] = (double)((i * 37) % 101); }
    cudaMalloc(&dx, N * sizeof(double)); cudaMalloc(&dy, N * sizeof(double));
    cudaMalloc(&dv, N * sizeof(double)); cudaMalloc(&dout, N / 64 * sizeof(double));
    cudaMemcpy(dx, hx, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv, N * sizeof(double), cudaMemcpyHostToDevice);
    run_saxpy();
    run_block_max();
    cudaMemcpy(hy, dy, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hout, dout, N / 64 * sizeof(double), cudaMemcpyDeviceToHost);
    double total = 0.0, best = 0.0;
    for (int i = 0; i < N; i++) total += hy[i];
    for (int i = 0; i < N / 64; i++) best = hout[i] > best ? hout[i] : best;
    printf("%.1f %.1f\n", total, best);
    timed(run_saxpy, "saxpy");
    timed(run_block_max, "block_max");
    cudaMemcpy(hy, dy, N * sizeof(double), cudaMemcpyDeviceToHost);
    timed(run_saxpy_copying, "saxpy with copies");
    timed(run_block_max_copying, "block_max with copies");
    return 0;
}
