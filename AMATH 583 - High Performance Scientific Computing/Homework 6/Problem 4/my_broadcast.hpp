# ifndef MY_BROADCAST_HPP
# define MY_BROADCAST_HPP

# include <mpi.h>
# include <iostream>
# include <vector>
# include <thread>
# include <cmath>
# include <stdexcept>
# include <tuple>
using namespace std;

template <typename T>
MPI_Datatype get_mpi_type();
template <> MPI_Datatype get_mpi_type<char>()               { return MPI_CHAR; }
template <> MPI_Datatype get_mpi_type<signed char>()        { return MPI_SIGNED_CHAR; }
template <> MPI_Datatype get_mpi_type<unsigned char>()      { return MPI_UNSIGNED_CHAR; }
template <> MPI_Datatype get_mpi_type<short>()              { return MPI_SHORT; }
template <> MPI_Datatype get_mpi_type<unsigned short>()     { return MPI_UNSIGNED_SHORT; }
template <> MPI_Datatype get_mpi_type<int>()                { return MPI_INT; }
template <> MPI_Datatype get_mpi_type<unsigned int>()       { return MPI_UNSIGNED; }
template <> MPI_Datatype get_mpi_type<long>()               { return MPI_LONG; }
template <> MPI_Datatype get_mpi_type<unsigned long>()      { return MPI_UNSIGNED_LONG; }
template <> MPI_Datatype get_mpi_type<long long>()          { return MPI_LONG_LONG; }
template <> MPI_Datatype get_mpi_type<unsigned long long>() { return MPI_UNSIGNED_LONG_LONG; }
template <> MPI_Datatype get_mpi_type<float>()              { return MPI_FLOAT; }
template <> MPI_Datatype get_mpi_type<double>()             { return MPI_DOUBLE; }
template <> MPI_Datatype get_mpi_type<long double>()        { return MPI_LONG_DOUBLE; }
template <> MPI_Datatype get_mpi_type<wchar_t>()            { return MPI_WCHAR; }

tuple<vector<double>, int, int, MPI_Comm> prepare_data(int i) {
    int count = i; // 8 * i bytes = i doubles.
    vector<double> buffer(count);
    for (int j = 0; j < count; ++j) {
        buffer[j] = i + j; // Fill with some test values.
    }
    return make_tuple(buffer, count, 0, MPI_COMM_WORLD);
}

tuple<vector<double>, int, MPI_Datatype, int, MPI_Comm> prepare_data_B(int i) {
    int count = i; // 8 * i bytes = i doubles
    vector<double> buffer(count);
    for (int j = 0; j < count; ++j) {
        buffer[j] = static_cast<double>(i + j); // Fill with test values
    }

    return make_tuple(buffer, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

template <typename T>
void my_broadcast(T* data, int count, int root, MPI_Comm comm) {
    // Get the current process number and the total number of processes.
    int proc_num, num_procs;
    MPI_Comm_rank(comm, &proc_num);
    MPI_Comm_size(comm, &num_procs);
    // Get data type.
    MPI_Datatype dtype = get_mpi_type<T>();

    // If this process is the root, send data to all other processes.
    if(proc_num == root) {
        for(int i=0; i < num_procs; i++) {
            if(i != root) {
                MPI_Send(data, count, dtype, i, 0, comm);
            }
        }
    }
    // If this process is not the root, receive the data from the root.
    else {
        MPI_Recv(data, count, dtype, root, 0, comm, MPI_STATUSES_IGNORE);
    }
}

#endif // ARC_LENGTH_HPP
