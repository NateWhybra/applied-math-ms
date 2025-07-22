# include <mpi.h>
# include <iostream>
# include <vector>
# include <thread>
# include <cmath>
# include <stdexcept>
# include <tuple>
# include "my_broadcast.hpp"
# include "timer.hpp"
using namespace std;

// // Get 4-PROC MY_BROADCAST.
// int main(int argc, char** argv) {    
//     // Initialize MPI.
//     MPI_Init(&argc, &argv); 

//     // Get the process number.
//     int proc_num;
//     MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);
 
//     string filename = "my_broadcast_4p";

//     // Wrap prepare_data and my_broadcast in lambdas so std::apply works in Timer.
//     auto prep = [](int i) {
//         return prepare_data(i); // tuple<vector<double>, int, int, MPI_Comm>
//     };

//     auto work = [](vector<double>& buffer, int count, int root, MPI_Comm comm) {
//         my_broadcast(buffer.data(), count, root, comm);
//         return 0.0; // Dummy return so time_work expects double.
//     };

//     // Call timer.
//     time_work(1, 32, filename, prep, work, (proc_num == 0), (proc_num == 0), proc_num);
    
//     // Finalize MPI.
//     MPI_Finalize();

//     return 0;
// }


// // Get 32-PROC MY_BROADCAST.
// int main(int argc, char** argv) {    
//     // Initialize MPI.
//     MPI_Init(&argc, &argv); 

//     // Get the process number.
//     int proc_num;
//     MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);
 
//     string filename = "my_broadcast_32p";

//     // Wrap prepare_data and my_broadcast in lambdas so std::apply works in Timer.
//     auto prep = [](int i) {
//         return prepare_data(i); // tuple<vector<double>, int, int, MPI_Comm>
//     };

//     auto work = [](vector<double>& buffer, int count, int root, MPI_Comm comm) {
//         my_broadcast(buffer.data(), count, root, comm);
//         return 0.0; // Dummy return so time_work expects double.
//     };

//     // Call timer.
//     time_work(1, 32, filename, prep, work, (proc_num == 0), (proc_num == 0), proc_num);
    
//     // Finalize MPI.
//     MPI_Finalize();

//     return 0;
// }


// // Get 4-PROC MPI_BROADCAST.
// int main(int argc, char** argv) {    
//     // Initialize MPI.
//     MPI_Init(&argc, &argv); 

//     // Get the process number.
//     int proc_num;
//     MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);
 
//     string filename = "mpi_4p";

//     // Wrap prepare_data_B and my_broadcast_B in lambdas so std::apply works in Timer.
//     auto prep = [](int i) {
//         return prepare_data_B(i); // tuple<vector<double>, int, MPI_Datatype, int, MPI_Comm>
//     };

//     auto work = [](std::vector<double>& buffer, int count, MPI_Datatype dtype, int root, MPI_Comm comm) {
//         MPI_Bcast(buffer.data(), count, dtype, root, comm);
//         return 0.0; // Dummy return so time_work expects double.
//     };

//     // Call timer.
//     time_work(1, 32, filename, prep, work, (proc_num == 0), (proc_num == 0), proc_num);
    
//     // Finalize MPI.
//     MPI_Finalize();

//     return 0;
// }

// Get 32-PROC MPI_BROADCAST.
int main(int argc, char** argv) {    
    // Initialize MPI.
    MPI_Init(&argc, &argv); 

    // Get the process number.
    int proc_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);
 
    string filename = "mpi_32p";

    // Wrap prepare_data_B and my_broadcast_B in lambdas so std::apply works in Timer.
    auto prep = [](int i) {
        return prepare_data_B(i); // tuple<vector<double>, int, MPI_Datatype, int, MPI_Comm>
    };

    auto work = [](std::vector<double>& buffer, int count, MPI_Datatype dtype, int root, MPI_Comm comm) {
        MPI_Bcast(buffer.data(), count, dtype, root, comm);
        return 0.0; // Dummy return so time_work expects double.
    };

    // Call timer.
    time_work(1, 32, filename, prep, work, (proc_num == 0), (proc_num == 0), proc_num);
    
    // Finalize MPI.
    MPI_Finalize();

    return 0;
}


