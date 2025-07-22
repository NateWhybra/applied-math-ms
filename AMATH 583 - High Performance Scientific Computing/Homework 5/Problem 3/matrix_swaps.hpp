# ifndef MATRIX_SWAPS_HPP
# define MATRIX_SWAPS_HPP
# include <fstream>
# include <stdexcept>
# include <vector>
# include <utility>

std::pair<int, int> getRandomIndices(int n) {
    int i = std::rand() % n;
    int j = std::rand() % (n - 1);
    if(j >= i) {
        j++;   
    }
    return std::make_pair(i, j);
};

// A is m x n
// A[i][j] = A[i + j * m] 
void swapRows(std::vector<double> &matrix, int nRows, int nCols, int i, int j) {
    // For each column value swap.
    for(int k = 0; k < nCols; k++) {
        // Grab value of i-th row, as a temp variable.
        double temp_i = matrix[i + k * nRows];
        
        // Swap.
        matrix[i + k * nRows] = matrix[j + k * nRows];
        matrix[j + k * nRows] = temp_i;
    }
};

void swapCols(std::vector<double> &matrix, int nRows, int nCols, int i, int j) {
    // For each row value, swap.
    for(int k = 0; k < nRows; k++) {
        // Grab value of i-th column, as a temp variable.
        double temp_i = matrix[k + i * nRows];
        
        // Swap.
        matrix[k + i * nRows] = matrix[k + j * nRows];
        matrix[k + j * nRows] = temp_i;
    }
};





# endif