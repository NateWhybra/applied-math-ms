# ifndef ARC_LENGTH_MPI_HPP
# define ARC_LENGTH_MPI_HPP

# include <mpi.h>
# include <iostream>
# include <vector>
# include <thread>
# include <cmath>
# include <stdexcept>
# include <tuple>
using namespace std;

// Prepare data for problem size i (A).
tuple<long> prepare_data_A(int i) {
    return make_tuple(1e8);
}

// Prepare data for problem size i (B).
tuple<long> prepare_data_B(int i) {
    return make_tuple(pow(10, i));
}

// Numbers used in integral calculation.
inline void arc_length(double a, double b, int num_points, double& result) {
    double sum = 0;
    double dx = (b - a) / num_points;
    double x = a;

    // Riemann sum (left end points).
    for (int i = 0; i < num_points; i++) {
        sum += ((1 / x) + (x / 4)) * dx;
        // Shift x.
        x += dx;
    }
    result = sum;
}

inline double split_calc_mpi(int num_points) {
    // Get curr_proc - The ID of the current processes.
    // Get num_proc - The number of processes.
    int curr_proc, num_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_proc);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    // Get interval width.
    const double a = 1.0, b = 6.0;
    double dx = (b - a) / num_points;

    // Compute global index range [i_start, i_end) for each process.
    int base_chunk = num_points / num_proc;
    int remainder = num_points % num_proc;
    int i_start = curr_proc * base_chunk + std::min(curr_proc, remainder);
    int i_end = i_start + base_chunk + (curr_proc < remainder ? 1 : 0);

    // Convert index range to subinterval [a_loc, b_loc].
    double a_loc = a + i_start * dx;
    double b_loc = a + i_end * dx;

    // Local number of intervals
    int num_loc = i_end - i_start;

    // Compute local Riemann sum
    double local_sum = 0.0;
    arc_length(a_loc, b_loc, num_loc, local_sum);

    // Reduce all local sums to global result on process 0.
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // If this is process 0, return global sum.
    if(curr_proc == 0) {
        return global_sum;
    }
    else {
        return 0.0;
    }
}


# endif // ARC_LENGTH_MPI_HPP