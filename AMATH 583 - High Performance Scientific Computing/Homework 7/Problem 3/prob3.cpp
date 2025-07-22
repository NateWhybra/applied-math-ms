# include <iostream>
# include <fstream>
# include <vector>
# include <chrono>
# include <cmath>
# include <cblas.h>
# include <cublas_v2.h>
# include <cuda_runtime.h>

void write_csv(const std::vector<int>& n_vals,
               const std::vector<double>& flops_openblas,
               const std::vector<double>& flops_cublas,
               const std::string& filename) {
    std::ofstream fout(filename);
    fout << "n,openblas,cublas\n";
    for (size_t i = 0; i < n_vals.size(); ++i)
        fout << n_vals[i] << "," << flops_openblas[i] << "," << flops_cublas[i] << "\n";
    fout.close();
}

// Returns FLOPs/s.
// Timer.
double benchmark_openblas(int n, int ntrial) {
    size_t size = n * n;
    std::vector<double> A(size), B(size), C(size);

    for (size_t i = 0; i < size; ++i) {
        A[i] = drand48();
        B[i] = drand48();
        C[i] = 0.0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < ntrial; ++t) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n,
                    1.0, A.data(), n,
                         B.data(), n,
                    1.0, C.data(), n);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double seconds = std::chrono::duration<double>(end - start).count() / ntrial;
    double flops = 2.0 * n * n * n / seconds;
    return flops;
}

// Timer.
double benchmark_cublas(int n, int ntrial) {
    size_t size = n * n;
    double *h_A = new double[size];
    double *h_B = new double[size];
    double *h_C = new double[size];

    for (size_t i = 0; i < size; ++i) {
        h_A[i] = drand48();
        h_B[i] = drand48();
        h_C[i] = 0.0;
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size * sizeof(double));
    cudaMalloc((void**)&d_B, size * sizeof(double));
    cudaMalloc((void**)&d_C, size * sizeof(double));

    cudaMemcpy(d_A, h_A, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 1.0, beta = 1.0;

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < ntrial; ++t) {
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n,
                    &alpha, d_A, n,
                            d_B, n,
                    &beta,  d_C, n);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double seconds = std::chrono::duration<double>(end - start).count() / ntrial;
    double flops = 2.0 * n * n * n / seconds;

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return flops;
}

int main() {
    std::vector<int> n_vals;
    std::vector<double> flops_openblas;
    std::vector<double> flops_cublas;

    const int ntrial = 3;
    for (int i = 1; i <= 13; ++i) {
        int n = 1 << i;  // n = 2, 4, ..., 8192
        std::cout << "Running n = " << n << std::endl;
        n_vals.push_back(n);

        double flop_open = benchmark_openblas(n, ntrial);
        double flop_cublas = benchmark_cublas(n, ntrial);
        flops_openblas.push_back(flop_open);
        flops_cublas.push_back(flop_cublas); 
    }

    write_csv(n_vals, flops_openblas, flops_cublas, "openblas_vs_cublas");

    return 0;
}
