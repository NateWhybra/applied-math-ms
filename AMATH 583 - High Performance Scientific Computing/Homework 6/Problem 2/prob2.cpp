# include <iostream>
# include <vector>
# include <thread>
# include <cmath>
# include <stdexcept>
# include "timer.hpp"
# include "arc_length_mpi.hpp"
# include "utils.hpp"
# include <tuple>
using namespace std;


// // For Part A.
// int main(int argc, char** argv) {
//     // Initialize MPI.
//     MPI_Init(&argc, &argv); 

//     // Get the process number.
//     int proc_num;
//     MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);

 
//     string filename = "problem_2";
//     time_work(1, 1, filename, prepare_data_A, split_calc_mpi, (proc_num == 0), (proc_num == 0), proc_num);
    
//     // Finalize MPI.
//     MPI_Finalize();

//     return 0;
// }

// For Part B.
int main(int argc, char** argv) {
    // Initialize MPI.
    MPI_Init(&argc, &argv); 

    // Get the process number.
    int proc_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);
 
    string filename = "problem_2_points";
    time_work(1, 6, filename, prepare_data_B, split_calc_mpi, (proc_num == 0), (proc_num == 0), proc_num);
    
    // Finalize MPI.
    MPI_Finalize();

    return 0;
}
