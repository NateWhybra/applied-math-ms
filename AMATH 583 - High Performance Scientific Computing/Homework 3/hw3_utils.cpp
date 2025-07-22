# include <vector>
# include <random>
# include <fstream>
# include "hw3_utils.hpp"
using namespace std;

vector<vector<double>> random_matrix(int m, int n, double low, double high) {
    // Make empty matrix of size m x n.
    vector<vector<double>> matrix(m, vector<double>(n));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(low, high);

    // Fill matrix with random values.
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i][j] = dist(gen);

    return matrix;
}

vector<double> random_vector(int n, double low, double high) {
    // Make empty vector of size n.
    vector<double> vec(n);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(low, high);

    // Fill vector with random values.
    for(int i = 0; i < n; i++) {
        vec[i] = dist(gen);
    }

    return vec;
}

void mat_to_csv(const vector<vector<double>> &A, const string &name) {
    // Make file name.
    string fname = name + ".csv";

    // Open file stream.
    ofstream fout(fname);
    
    // Write the matrix.
    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[i].size(); ++j) {
            fout << A[i][j];
            if (j != A[i].size() - 1) fout << ",";
        }
        fout << "\n";
    }
    
    // Close file stream.
    fout.close();
}

void vec_to_csv(const vector<double> &v, const string &name) {
    // Make file name.
    string fname = name + ".csv";

    // Open file stream.
    ofstream fout(fname);
    
    // Write the vector.
    for (int i = 0; i < v.size(); i++) {
        fout << v[i];
        if (i != v.size() - 1) fout << ",";
    }
    
    // Close file stream.
    fout.close();
}