# include <vector>
# include <random>
# include <string>
# include <fstream>
# include <stdexcept>
# include "utils.hpp"
# include <chrono>
# include <iostream>
using namespace std;


void write_to_binary(vector<double> A, string save_file_name) {
    // Open file in binary mode.
    ofstream out(save_file_name, ios::binary);
    
    if (!out) {
        throw runtime_error("Failed to open file for writing: " + save_file_name);
    }

    // Optionally write the size of the vector first (useful for later reading).
    size_t size = A.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size_t));

    // Write the actual vector data.
    out.write(reinterpret_cast<const char*>(A.data()), size * sizeof(double));

    out.close();
}


vector<double> read_from_binary(const string& file_name) {
    ifstream in(file_name, ios::binary);
    
    if (!in) {
        throw runtime_error("Failed to open file for reading: " + file_name);
    }

    // Read the size of the vector first.
    size_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(size_t));

    // Read the actual vector data.
    vector<double> A(size);
    in.read(reinterpret_cast<char*>(A.data()), size * sizeof(double));

    in.close();
    return A;
}

int main() {
    // Timer foo.
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    long double elapsed_time = 0.L;
    long double avg_time;
    const int ntrials = 3;
    vector<double> results_read(9);
    vector<double> results_write(9);
    int n = 0;
    string file_name = "";
    
    // Loop on problem size.
    for(int i = 5; i <= 13; i++) {
        // n for matrix size.
        n = 1 << i;

        // Define a vector of length n. This is our matrix in "column major order".
        // Let's just say we have an (n x 1 matrix).
        long dim = n * n;

        cout << dim << endl;
        vector<double> A = random_vector(dim, 0.0, 1.0);
            
        // Perform an experiment.
        for (int t = 0; t < ntrials; t++) {
            start = std::chrono::high_resolution_clock::now();     
            
            // Write.
            file_name = to_string(n);
            write_to_binary(A, file_name);
            
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            elapsed_time += (duration.count() * 1.e-9); // Convert duration to seconds.
        }
        avg_time = elapsed_time / static_cast<long double>(ntrials);

        // Print.
        cout << "Problem size = " << n << ", Avg time = " << avg_time << " seconds." << endl;
        
        // Save or report findings.
        results_write[i-5] = avg_time;
        
        // Zero time again.
        elapsed_time = 0.L;
    }
    cout << " " << endl;

    // Name for results.
    file_name = "write_results";
    // Save data to .csv.
    vec_to_csv(results_write, file_name);

    // Loop on problem size.
    for(int i = 5; i <= 13; i++) {
        // n for matrix size.
        n = 1 << i;
            
        // Perform an experiment.
        for (int t = 0; t < ntrials; t++) {
            start = std::chrono::high_resolution_clock::now();     
            
            // Read.
            file_name = to_string(n);
            vector<double> B = read_from_binary(file_name);
            
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            elapsed_time += (duration.count() * 1.e-9); // Convert duration to seconds.
        }
        avg_time = elapsed_time / static_cast<long double>(ntrials);

        // Print.
        cout << "Problem size = " << n << ", Avg time = " << avg_time << " seconds." << endl;
        
        // Save or report findings.
        results_read[i-5] = avg_time;
        
        // Zero time again.
        elapsed_time = 0.L;
    }
    cout << " " << endl;

    // Name for results.
    file_name = "read_results";
    // Save data to .csv.
    vec_to_csv(results_read, file_name);
}

