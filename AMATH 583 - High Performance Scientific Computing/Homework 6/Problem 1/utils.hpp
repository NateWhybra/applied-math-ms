# ifndef UTILS_HPP
# define UTILS_HPP
# include <vector>
# include <random>
# include <string>
# include <fstream>

template <typename T>
std::vector<std::vector<T>> random_matrix(int m, int n, T low, T high) {
    // Make empty matrix of size m x n.
    std::vector<std::vector<T>> matrix(m, std::vector<T>(n));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(low, high);

    // Fill matrix with random values.
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i][j] = T(dist(gen));

    return matrix;
};

template <typename T>
std::vector<T> random_vector(long n, T low, T high) {
    // Make empty vector of size n.
    std::vector<T> vec(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(low, high);

    // Fill vector with random values.
    for(int i = 0; i < n; i++) {
        vec[i] = T(dist(gen));
    }

    return vec;
};

template <typename T>
void mat_to_csv(const std::vector<std::vector<T>> &A, const std::string &name) {
    // Make file name.
    std::string fname = name + ".csv";

    // Open file stream.
    std::ofstream fout(fname);
    
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

template <typename T>
void vec_to_csv(const std::vector<T> &v, const std::string &name) {
    // Make file name.
    std::string fname = name + ".csv";

    // Open file stream.
    std::ofstream fout(fname);
    
    // Write the vector.
    for (int i = 0; i < v.size(); i++) {
        fout << v[i];
        if (i != v.size() - 1) fout << ",";
    }
    
    // Close file stream.
    fout.close();
}

# endif