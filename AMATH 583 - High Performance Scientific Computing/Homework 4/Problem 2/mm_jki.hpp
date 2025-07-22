# ifndef JKI_HPP
# define JKI_HPP
# include <vector>

template <typename T>
void mm_jki(T a, const std::vector<T> &A, const std::vector<T> &B , T b , std::vector<T> &C, int m, int p, int n) {
    // Assume row major order.

    // A is m x p.
    // B is p x n.
    // C is m x n.

    // Scale C by b.
    for(int i = 0; i < m * n; i++) {
        C[i] *= b;
    }

    // (j outer, k middle, i outer)
    for(int j = 0; j < n; j++) {
        for(int k = 0; k < p; k++) {
            // Every element of the j^th column of B needs to to be multiplied with A[i][k] and summed over n.
            // T r = B[k][j];
            T r = B[k * n + j];
            for(int i = 0; i < m; i++) {
                // C[i][j] += a * A[i][k] * B[k][j] 
                C[i * n + j] += a * A[i * p + k] * r;
            }
        }
    }
};

# endif