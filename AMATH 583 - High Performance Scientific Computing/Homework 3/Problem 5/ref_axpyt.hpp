# ifndef AXPY_HPP
# define AXPY_HPP
# include <vector>
# include <stdexcept>

template<typename T>
void axpy(T a, const std::vector<T> &x, std::vector<T> &y) {
    // Check that the vectors are the same size.
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vectors must be the same length.");
    }
    
    // Otherwise, do the daxpy.
    for(int i = 0; i < x.size(); i++) {
        y[i] += a * x[i];
    }
}

# endif