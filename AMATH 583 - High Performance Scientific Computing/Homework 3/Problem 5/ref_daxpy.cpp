# include <iostream>
# include <vector>
# include <stdexcept>
# include <chrono>
# include "hw3_utils.hpp"
using namespace std;

void daxpy(double a, const vector<double> &x, vector<double> &y) {
    // Check that the vectors are the same size.
    if (x.size() != y.size()) {
        throw invalid_argument("Vectors must be the same length.");
    }
    
    // Otherwise, do the daxpy.
    for(int i = 0; i < x.size(); i++) {
        y[i] += a * x[i];
    }
}

