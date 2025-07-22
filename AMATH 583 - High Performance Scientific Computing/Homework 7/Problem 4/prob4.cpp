#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

// Chck if Cuda hasn't caught an error.
void check_cuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " — " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int ntrial = 10;
    std::vector<size_t> sizes;
    std::vector<double> h2d_bandwidth;
    std::vector<double> d2h_bandwidth;

    // Test from 1 byte up to 2 GB.
    for (int i = 0; i <= 31; ++i) {
        size_t size = 1ULL << i;
        sizes.push_back(size);

        char* h_data;
        char* d_data;
        check_cuda(cudaMallocHost(&h_data, size), "cudaMallocHost");
        check_cuda(cudaMalloc(&d_data, size), "cudaMalloc");

        // Warm-up.
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
        cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

        // H2D timing.
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < ntrial; ++t) {
            cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double h2d_time = std::chrono::duration<double>(end - start).count() / ntrial;
        h2d_bandwidth.push_back(static_cast<double>(size) / h2d_time);

        // D2H timing.
        cudaDeviceSynchronize();
        start = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < ntrial; ++t) {
            cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        double d2h_time = std::chrono::duration<double>(end - start).count() / ntrial;
        d2h_bandwidth.push_back(static_cast<double>(size) / d2h_time);

        cudaFree(d_data);
        cudaFreeHost(h_data);
    }

    // Write CSV.
    std::ofstream fout("gpu_copy_bandwidth.csv");
    fout << "bytes,h2d_bandwidth,d2h_bandwidth\n";
    for (size_t i = 0; i < sizes.size(); ++i) {
        fout << sizes[i] << "," << h2d_bandwidth[i] << "," << d2h_bandwidth[i] << "\n";
    }
    fout.close();

    std::cout << "Benchmark complete. Results written to gpu_copy_bandwidth.csv.\n";
    return 0;
}
