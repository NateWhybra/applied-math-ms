# ifndef GEMV_HPP
# define GEMV_HPP
# include <vector>
# include <stdexcept>

template<typename T>
void gemv(T a, const std::vector<std::vector<T>> &A, const std::vector<T> &x, T b, std::vector<T> &y) {
    // A is m x n and x is n x 1.
    if (A[0].size() != x.size()) {
       throw std::invalid_argument("Matrix A and Vector x have incompatible dimensons for multiplication.");
   }
   // A is m x n and y is m x 1.
   if (A.size() != y.size()) {
       throw std::invalid_argument("Matrix A and Vector y have incompatible dimensons for multiplication.");
   }

   // Initialize temporary variable.
   T current_val = T(0.0);
   for(int i = 0; i < A.size(); i++) { 
       for(int j = 0; j < A[0].size(); j++) {
           current_val += A[i][j] * x[j]; 
       }
       y[i] = a * current_val + b * y[i];
       current_val = T(0.0);
   }
}

# endif