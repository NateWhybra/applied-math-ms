// nwhybra@uw.edu
// AMATH 483-583
// strassen.cpp : starter code for Strassen implementation.

# include <iostream>
# include <vector>
# include <chrono>
# include "utils.hpp"
# include <bit>
using namespace std;

template <typename T>
vector<vector<T>> addMatrix(const vector<vector<T>> &A, const vector<vector<T>> &B)
{
    int n = A.size();
    int m = A[0].size();
    vector<vector<T>> C(n, vector<T>(m));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

template <typename T>
vector<vector<T>> subtractMatrix(const vector<vector<T>> &A, const vector<vector<T>> &B)
{
    int n = A.size();
    int m = A[0].size();
    vector<vector<T>> C(n, vector<T>(m));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

template <typename T>
vector<vector<T>> naiveMultiply(const vector<vector<T>> &A, const vector<vector<T>> &B) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<T>> C(n, vector<T>(m, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

template <typename T>
vector<vector<T>> strassenMultiply(const vector<vector<T>> &A, const vector<vector<T>> &B)
{
    // Get the matrix dimensions, just assume A and B are the same size.
    int m = A.size();
    int n = A[0].size();

    // Make space for output.
    vector<vector<T>> C(m, vector<T>(n));

    // Base case (used regular matrix multiplication).
    if(m <= 8) {
        C = naiveMultiply(A, B);
        return C;
    }
    else {
        // Split A and B into 4 sub-matrices.
        int k = m / 2;

        // Split A.
        vector<vector<T>> A_11(k, vector<T>(k));
        vector<vector<T>> A_12(k, vector<T>(k));
        vector<vector<T>> A_21(k, vector<T>(k));
        vector<vector<T>> A_22(k, vector<T>(k));

        // Split B.
        vector<vector<T>> B_11(k, vector<T>(k));
        vector<vector<T>> B_12(k, vector<T>(k));
        vector<vector<T>> B_21(k, vector<T>(k));
        vector<vector<T>> B_22(k, vector<T>(k));

        // Actually do the splitting.
        for(int i = 0; i < k; i++) {
            for(int j = 0; j < k; j++) {
                // Top left block.
                A_11[i][j] = A[i][j];
                B_11[i][j] = B[i][j];

                // Top right block,
                A_12[i][j] = A[i][j + k];
                B_12[i][j] = B[i][j + k];

                // Bottom left block.
                A_21[i][j] = A[i + k][j];
                B_21[i][j] = B[i + k][j];

                // Bottom right block.
                A_22[i][j] = A[i + k][j + k];
                B_22[i][j] = B[i + k][j + k];
            }
        }

        // Do Strassen on each piece.
        vector<vector<T>> m1 = strassenMultiply(addMatrix(A_11, A_22), addMatrix(B_11, B_22));
        vector<vector<T>> m2 = strassenMultiply(addMatrix(A_21, A_22), B_11);
        vector<vector<T>> m3 = strassenMultiply(A_11, subtractMatrix(B_12, B_22));
        vector<vector<T>> m4 = strassenMultiply(A_22, subtractMatrix(B_21, B_11));
        vector<vector<T>> m5 = strassenMultiply(addMatrix(A_11, A_12), B_22);
        vector<vector<T>> m6 = strassenMultiply(subtractMatrix(A_21, A_11), addMatrix(B_11, B_12));
        vector<vector<T>> m7 = strassenMultiply(subtractMatrix(A_12, A_22), addMatrix(B_21, B_22));

        // Combine resulting matrix (fill C).
        // Actually do the splitting.
        for(int i = 0; i < k; i++) {
            for(int j = 0; j < k; j++) {
                // Top left block.
                C[i][j] = m1[i][j] + m4[i][j] - m5[i][j] + m7[i][j];

                // Top right block,
                C[i][j + k] = m3[i][j] + m5[i][j];

                // Bottom left block.
                C[i + k][j] = m2[i][j] + m4[i][j];

                // Bottom right block.
                C[i + k][j + k] = m1[i][j] - m2[i][j] + m3[i][j] + m6[i][j];
            }
        }

        return C;
    }
}

template <typename T>
void printMatrix(const vector<vector<T>> &matrix)
{
    int n = matrix.size();
    int m = matrix[0].size();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

// Find the next power of 2.
int nextPowerOfTwo(int x) {
    if (x <= 0) return 1;
    int power = 1;
    while (power < x) {
        power <<= 1;  // Multiply by 2.
    }
    return power;
}

template <typename T>
vector<vector<T>> pad(vector<vector<T>> A, int new_size) {
    // Make storage for new matrix.
    vector<vector<T>> B(new_size, vector<T>(new_size));

    // Fill new matrix.
    for(int i = 0; i < A.size(); i++) {
        for(int j = 0; j < A[0].size(); j++) {
            B[i][j] = A[i][j];
        }
    }

    return B;
}


// int
template vector<vector<int>> addMatrix<int>(const vector<vector<int>> &A, const vector<vector<int>> &B);
template vector<vector<int>> subtractMatrix<int>(const vector<vector<int>> &A, const vector<vector<int>> &B);
template vector<vector<int>> strassenMultiply<int>(const vector<vector<int>> &A, const vector<vector<int>> &B);
template void printMatrix<int>(const vector<vector<int>> &matrix);
// double
template vector<vector<double>> addMatrix<double>(const vector<vector<double>> &A, const vector<vector<double>> &B);
template vector<vector<double>> subtractMatrix<double>(const vector<vector<double>> &A, const vector<vector<double>> &B);
template vector<vector<double>> strassenMultiply<double>(const vector<vector<double>> &A, const vector<vector<double>> &B);
template void printMatrix<double>(const vector<vector<double>> &matrix);

int main() {
    // Timer foo.
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time = 0.L;
    long double avg_time;
    const int ntrials = 3;
    string file_name = "";
    vector<double> results(256);

    // Loop on problem size.
    for(int i = 1; i <= 256; i++) {
        // Make n even.
        int n = 2 * i;

        // Find next power of 2.
        int next_pow = nextPowerOfTwo(n);

        // Make a random matrix in column major order.
        vector<vector<double>> A = random_matrix(n, n, 0.0, 1.0);
        vector<vector<double>> B = random_matrix(n, n, 0.0, 1.0);

        // Pad A and B.
        A = pad(A, next_pow);
        B = pad(B, next_pow);

        // Perform row swap experiment.
        for(int t = 0; t < ntrials; t++) {
            start = std::chrono::high_resolution_clock::now();     
            // Do work.
            vector<vector<double>> C = strassenMultiply(A, B);
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            elapsed_time += (duration.count() * 1.e-9); // Convert duration to seconds.
        }
        avg_time = elapsed_time / static_cast<long double>(ntrials);

        cout << i << endl;
        cout << avg_time << endl;
        
        // Save or report findings.
        results[i-1] = avg_time;
        
        // Zero time again.
        elapsed_time = 0.L;
    }

    // Name for results.
    file_name = "results_strassen";

    // Save data to .csv.
    vec_to_csv(results, file_name);
}
