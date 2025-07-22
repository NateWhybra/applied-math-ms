#ifndef FILE_SWAPS_HPP
#define FILE_SWAPS_HPP
#include <fstream>
#include <stdexcept>
#include <vector>

void swapRowsInFile(std::fstream &file, int nRows, int nCols, int i, int j) {
    // Don't do anything.
    if (i == j) return;

    // Error checks.
    if (i < 0 || i >= nRows || j < 0 || j >= nRows)
        throw std::invalid_argument("Row indices out of bounds.");

    // Clear flags in case of earlier EOF or fail.
    file.clear();

    // For swapping values.
    double temp1, temp2;

    // Offset to skip the size_t header at the start of the file.
    std::streamoff offset = sizeof(std::size_t);

    // For each column, swap the elements at each row index.
    for (int col = 0; col < nCols; ++col) {
        // Row i at column col is at offset (col * nRows + i)
        // Row j at column col is at offset (col * nRows + j)
        std::streampos pos_i = offset + (col * nRows + i) * sizeof(double);
        std::streampos pos_j = offset + (col * nRows + j) * sizeof(double);

        // Read value at (i, col).
        file.seekg(pos_i);
        file.read(reinterpret_cast<char*>(&temp1), sizeof(double));

        // Read value at (j, col).
        file.seekg(pos_j);
        file.read(reinterpret_cast<char*>(&temp2), sizeof(double));

        // Write temp2 to (i, col).
        file.seekp(pos_i);
        file.write(reinterpret_cast<char*>(&temp2), sizeof(double));

        // Write temp1 to (j, col).
        file.seekp(pos_j);
        file.write(reinterpret_cast<char*>(&temp1), sizeof(double));
    }
}

void swapColsInFile(std::fstream &file, int nRows, int nCols, int i, int j) {
    // Don't do anything.
    if (i == j) return;

    // Error checks.
    if (i < 0 || i >= nCols || j < 0 || j >= nCols)
        throw std::invalid_argument("Column indices out of bounds.");

    // Clear flags in case of earlier EOF or fail.
    file.clear();

    // Make temporary storage for the swapped columns.
    std::vector<double> col_i(nRows), col_j(nRows);

    // Offset to skip the size_t header at the start of the file.
    std::streamoff offset = sizeof(std::size_t);

    // Read entire column i.
    file.seekg(offset + i * nRows * sizeof(double));
    file.read(reinterpret_cast<char*>(col_i.data()), nRows * sizeof(double));

    // Read entire column j.
    file.seekg(offset + j * nRows * sizeof(double));
    file.read(reinterpret_cast<char*>(col_j.data()), nRows * sizeof(double));

    // Write column j to column i.
    file.seekp(offset + i * nRows * sizeof(double));
    file.write(reinterpret_cast<char*>(col_j.data()), nRows * sizeof(double));

    // Write column i to column j.
    file.seekp(offset + j * nRows * sizeof(double));
    file.write(reinterpret_cast<char*>(col_i.data()), nRows * sizeof(double));
}

void write_to_binary(std::vector<double> A, std::string save_file_name) {
    // Open file in binary mode.
    std::ofstream out(save_file_name, std::ios::binary);
    
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + save_file_name);
    }

    // Optionally write the size of the vector first (useful for later reading).
    std::size_t size = A.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(std::size_t));

    // Write the actual vector data.
    out.write(reinterpret_cast<const char*>(A.data()), size * sizeof(double));

    out.close();
}

std::vector<double> read_from_binary(const std::string& file_name) {
    std::ifstream in(file_name, std::ios::binary);
    
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + file_name);
    }

    // Read the size of the vector first.
    std::size_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));

    // Read the actual vector data.
    std::vector<double> A(size);
    in.read(reinterpret_cast<char*>(A.data()), size * sizeof(double));

    in.close();
    return A;
}

#endif