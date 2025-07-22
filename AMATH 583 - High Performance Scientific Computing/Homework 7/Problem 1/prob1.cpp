# include <cblas.h> 
# include <iostream>
# include "timer.hpp"
# include "utils.hpp"
# include <vector>
# include <chrono>
using namespace std;

// Helper functions for timer.
tuple<int, double, vector<double>, int, vector<double>, int> pd_daxpy(int n) {
    vector<double> X(2 * n);
    vector<double> Y(2 * n);
    double alpha = 1;
    return make_tuple(2*n, alpha, X, 1, Y, 1);
}

double daxpy(int N, double alpha, const std::vector<double>& X, int incX, std::vector<double>& Y, int incY) {
    cblas_daxpy(N, alpha, X.data(), incX, Y.data(), incY);
    return 0;
}

tuple<int, double, vector<double>, vector<double>, vector<double>, double> pd_dgemv(int n) {
    vector<double> A(2*n * 2*n);  // Col-major
    vector<double> X(2*n);
    vector<double> Y(2*n);
    double alpha = 1.0, beta = 1.0;
    return make_tuple(2*n, alpha, A, X, Y, beta);
}

double dgemv(int N, double alpha, const vector<double>& A, const vector<double>& X, vector<double>& Y, double beta) {
    cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, alpha, A.data(), N, X.data(), 1, beta, Y.data(), 1);
    return 0; 
}

tuple<int, double, vector<double>, vector<double>, vector<double>, double> pd_dgemm(int n) {
    vector<double> A(2*n * 2*n), B(2*n * 2*n), C(2*n * 2*n);
    double alpha = 1.0, beta = 1.0;
    return make_tuple(2*n, alpha, A, B, C, beta);
}

double dgemm(int N, double alpha, const vector<double>& A, const vector<double>& B, vector<double>& C, double beta) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
    return 0;
}

int main() {
    // Check if working.
    cout << "Running!" << endl;

    // Time cblas_daxpy.
    auto results_daxpy = time_work(3, 2048, "cblas_daxpy", pd_daxpy, daxpy, true, false);

    // Time cblas_dgemv.
    auto results_dgemv = time_work(3, 2048, "cblas_dgemv", pd_dgemv, dgemv, true, false);

    // Time cblas_dgemm.
    auto results_dgemm = time_work(3, 2048, "cblas_dgemm", pd_dgemm, dgemm, true, false);

    return 0;
}