# ifndef KIJ_HPP
# define KIJ_HPP
# include <vector>

template <typename T>
void mm_kij(T a, const std::vector<T> &A, const std::vector<T> &B , T b, std::vector<T> &C, int m, int p, int n) {
    // Assume row major order.

    // Scale C by b.
    for(int i = 0; i < m * n; i++) {
        C[i] *= b;
    }

    // (k outer, i middle, j outer)
    for(int k = 0; k < p; k++) {
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                C[i * n + j] += a * A[i * p + k] * B[k * n + j];
            }
        }
    }
};

# endif