# include <vector>
# include <random>
# include <string>
# include <fstream>
# include <stdexcept>
# include "utils.hpp"
# include "file_swaps.hpp"
# include <chrono>
# include <iostream>
using namespace std;

int main() {
    // Timer foo.
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time = 0.L;
    long double avg_time;
    const int ntrials = 50;
    vector<double> results_row(9);
    vector<double> results_col(9);
    int n = 0;
    string file_name = "";
    
    // Loop on problem size.
    for(int i = 5; i <= 13; i++) {
        // n for matrix size.
        n = 1 << i;

        // Define a vector of length n x n. This is our matrix in "column major order".
        vector<double> A = random_vector(n * n, 0.0, 1.0);
        
        // Write.
        file_name = to_string(n);
        write_to_binary(A, file_name);

        // Open the file.
        // Open the file in read/write binary mode
        std::fstream file(file_name, std::ios::in | std::ios::out | std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Failed to open file.\n";
            return 1;
        }

        // Swap rows 0 and 3.
        // Perform an experiment.
        for (int t = 0; t < ntrials; t++) {
            start = std::chrono::high_resolution_clock::now();     
            
            swapRowsInFile(file, n, n, 0, 3);

            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            elapsed_time += (duration.count() * 1.e-9); // Convert duration to seconds.
        }
        avg_time = elapsed_time / static_cast<long double>(ntrials);

        // Print.
        cout << "Problem size = " << n << ", Avg time = " << avg_time << " seconds." << endl;
        
        // Save or report findings.
        results_row[i-5] = avg_time;
        
        // Zero time again.
        elapsed_time = 0.L;
    }
    cout << " " << endl;

    // Name for results.
    file_name = "row_results";
    // Save data to .csv.
    vec_to_csv(results_row, file_name);

    // Loop on problem size.
    for(int i = 5; i <= 13; i++) {
        // n for matrix size.
        n = 1 << i;

        // Define a vector of length n x n. This is our matrix in "column major order".
        vector<double> A = random_vector(n * n, 0.0, 1.0);
        
        // Open the file in read/write binary mode.
        file_name = to_string(n);
        std::fstream file(file_name, std::ios::in | std::ios::out | std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Failed to open file.\n";
            return 1;
        }

        // Swap cols 0 and 3.
        // Perform an experiment.
        for (int t = 0; t < ntrials; t++) {
            start = std::chrono::high_resolution_clock::now();     
            
            swapColsInFile(file, n, n, 0, 3);

            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            elapsed_time += (duration.count() * 1.e-9); // Convert duration to seconds.
        }
        avg_time = elapsed_time / static_cast<long double>(ntrials);

        // Print.
        cout << "Problem size = " << n << ", Avg time = " << avg_time << " seconds." << endl;
        
        // Save or report findings.
        results_col[i-5] = avg_time;
        
        // Zero time again.
        elapsed_time = 0.L;
    }
    cout << " " << endl;

    // Name for results.
    file_name = "col_results";
    // Save data to .csv.
    vec_to_csv(results_col, file_name);
}
