# include <iostream>
# include <fstream>
# include <complex>
# include <cstdlib>
# include <cmath>
# include <vector>
# include <limits>
# include <cblas.h>
# include <lapacke.h>
# include "utils.hpp"

int main() {
    std::vector<int> n_vals;
    std::vector<double> residuals;
    std::vector<double> norm_errors;
    double eps = std::numeric_limits<double>::epsilon();

    for (int idx = 0; idx < 10; ++idx) {
        int n = 1 << (idx + 4);  // 16, 32, ..., 8192.
        n_vals.push_back(n);

        // Allocate memory.
        std::complex<double>* a = new std::complex<double>[n * n];
        std::complex<double>* b = new std::complex<double>[n];
        std::complex<double>* b_orig = new std::complex<double>[n];
        std::complex<double>* Az = new std::complex<double>[n];

        // Generate A (diagonally dominant).
        srand(0);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                double real = 0.5 - (double)rand() / RAND_MAX;
                double imag = 0.5 - (double)rand() / RAND_MAX;
                std::complex<double> val(real, imag);
                if (i == j) val *= n;
                a[i + j * n] = val;
            }
        }

        // Generate random b.
        srand(1);
        for (int i = 0; i < n; ++i) {
            double real = 0.5 - (double)rand() / RAND_MAX;
            double imag = 0.5 - (double)rand() / RAND_MAX;
            b[i] = std::complex<double>(real, imag);
            b_orig[i] = b[i];  // Save original b.
        }

        // Solve A z = b using LAPACK.
        std::vector<lapack_int> ipiv(n);
        lapack_int info = LAPACKE_zgesv(
            LAPACK_COL_MAJOR, n, 1,
            reinterpret_cast<lapack_complex_double*>(a), n,
            ipiv.data(),
            reinterpret_cast<lapack_complex_double*>(b), n
        );

        if (info != 0) {
            std::cerr << "LAPACKE_zgesv failed at n = " << n << ", info = " << info << std::endl;
            return 1;
        }

        // b now contains z_hat; compute A * z_hat.
        std::complex<double> one(1.0, 0.0), zero(0.0, 0.0);
        cblas_zgemv(CblasColMajor, CblasNoTrans, n, n,
                    &one, reinterpret_cast<void*>(a), n,
                    reinterpret_cast<void*>(b), 1,
                    &zero, reinterpret_cast<void*>(Az), 1);

        // Compute r = A z_hat - b_orig.
        for (int i = 0; i < n; ++i)
            Az[i] -= b_orig[i];

        // ||A z_hat - b||_2
        double residual = 0.0;
        for (int i = 0; i < n; ++i)
            residual += std::norm(Az[i]);
        residual = std::sqrt(residual);

        // ||z_hat||_2 (z_hat is stored in b)
        double z2 = 0.0;
        for (int i = 0; i < n; ++i)
            z2 += std::norm(b[i]);
        z2 = std::sqrt(z2);

        // ||A||_1 = max column sum
        double A1 = 0.0;
        for (int j = 0; j < n; ++j) {
            double col_sum = 0.0;
            for (int i = 0; i < n; ++i)
                col_sum += std::abs(a[i + j * n]);
            A1 = std::max(A1, col_sum);
        }

        // Normalized error.
        double norm_err = residual / (A1 * z2 * eps);

        residuals.push_back(residual);
        norm_errors.push_back(norm_err);

        std::cout << "n = " << n
                  << ", residual = " << residual
                  << ", normalized error = " << norm_err << std::endl;

        delete[] a;
        delete[] b;
        delete[] b_orig;
        delete[] Az;
    }

    vec_to_csv(residuals, "res");
    vec_to_csv(norm_errors, "errs");

    return 0;
}