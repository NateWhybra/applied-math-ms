# ifndef GEMM_HPP
# define GEMM_HPP
# include <vector>
# include <stdexcept>

template<typename T> 
void gemm(T a, const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B, T b, 
    std::vector<std::vector<T>> &C) {
       
    // A is m x p and B is p x n.
    // Check if A and B are compatible.
    if (A[0].size() != B.size()) {
       throw std::invalid_argument("Matrix A and Matrix B have incompatible dimensions.");
   }
   // Check if (A, B) and C are compatible.
   if (A.size() != C.size())  {
       throw std::invalid_argument("Matrix A and Matrix C have incompatible dimensions.");
   }
   if (B[0].size() != C[0].size()) {
       throw std::invalid_argument("Matrix B and Matrix C have incompatible dimensions.");
   }

   // For each row of C.
   for(int i = 0; i < C.size(); i++) { 
       // For each column of C.
       for(int j = 0; j < C[0].size(); j++) {
           // Initialize temporary variable.
           T current_val = T(0.0);
           // Compute the matrix product AB[i, j].
           for(int k = 0; k < A[0].size(); k++) {
               current_val += A[i][k] * B[k][j];
           } 
           // C = a*(AB) + b*(C).
           C[i][j] = a * current_val + b * C[i][j];
       }
   }
}

# endif