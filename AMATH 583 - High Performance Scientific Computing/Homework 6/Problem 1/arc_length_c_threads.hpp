# ifndef ARC_LENGTH_CT_HPP
# define ARC_LENGTH_CT_HPP

# include <iostream>
# include <vector>
# include <thread>
# include <cmath>
# include <stdexcept>
using namespace std;

// Prepare data for problem size i (A).
// (num_points, num_threads)
std::pair<long, long> prepare_data_A(int i) {
    return {1e8, pow(2, i-1)};
}

// Prepare data for problem size i (B).
// (num_points, num_threads)
std::pair<long, long> prepare_data_B(int i) {
    return {pow(10, i), 1};
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

inline double split_calc(int num_points, int num_threads) {
    // Count how many threads there are.
    int real_num_threads = thread::hardware_concurrency();

    // Throw warning if asking for too many threads.
    if (num_threads > real_num_threads) {
        throw invalid_argument("Your CPU does not have the number of threads you requested. Try again.");
    }

    // Split the interval into num_threads intervals.
    vector<double> a_vals(num_threads);
    vector<double> b_vals(num_threads);

    double a = 1;
    double b = 6;
    double width = (b - a) / num_threads;

    for (int i = 0; i < num_threads; i++) {
        a_vals[i] = a;
        b_vals[i] = a + width;
        a += width;
    }

    // Number of points in each interval.
    int num_points_each = num_points / num_threads;

    // Make a vector to store threads and results.
    vector<thread> threads(num_threads);
    vector<double> results(num_threads);

    // For each thread, give it a sub-integral to compute.
    for (int i = 0; i < num_threads; i++) {
        threads[i] = thread(arc_length, a_vals[i], b_vals[i], num_points_each, ref(results[i]));
    }

    // Tell main thread to wait for the sub-threads to be done.
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }

    // Return the sum of the results.
    double result = 0;
    for (int i = 0; i < num_threads; i++) {
        result += results[i];
    }

    return result;
}

#endif // ARC_LENGTH_CT_HPP