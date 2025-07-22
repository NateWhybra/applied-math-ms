# ifndef TIMER_HPP
# define TIMER_HPP

# include <vector>
# include <string>
# include <chrono>
# include <functional>
# include <iostream>
# include <iomanip>
# include "utils.hpp"
// for std::forward
# include <utility>
# include <tuple>

// Template benchmarking function.

// num_trials = number of times for each problem size
// max_size = maximum problem size (i in [1, ..., max_size])
// filename = save file name for .csv file.

// prepare_data = a function that takes a single integer as input and returns a tuple of objects.
// This function prepares the input data for the work function.

// do_work = function that takes input from prepare_data, and performs the actual work we want timed. Right now, we assume do work outputs a double.
template <typename PrepareFunc, typename WorkFunc>
std::pair<std::vector<double>, std::vector<double>> time_work(int num_trials, int max_size, const std::string& filename, PrepareFunc prepare_data, WorkFunc do_work, bool save_data, bool append, int proc_num=0) {
    using Clock = std::chrono::high_resolution_clock;

    std::vector<double> time_results(max_size);
    std::vector<double> work_results(max_size);
    long double elapsed_time = 0.0;
    
    for(int i = 1; i <= max_size; i++) {
        // Prepare inputs for problem size i.
        auto input_args = prepare_data(i); // input_args must be a tuple.

        // Define result as the type of data outputed by work function.
        decltype(std::apply(do_work, input_args)) last_result;

        for (int t = 0; t < num_trials; ++t) {
            auto start = Clock::now();

            // Unpack the tuple and call do_work(args...)...
            last_result = std::apply(do_work, input_args);

            auto stop = Clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            elapsed_time += duration.count() * 1e-9;
        }

        if(proc_num == 0) {
            long double avg_time = elapsed_time / num_trials;
            time_results[i - 1] = avg_time;
            // Assuming for now that the function outputs doubles.
            work_results[i - 1] = static_cast<double>(last_result); 
            // std::cout << "Size = " << i << ", Avg Time = " << std::setprecision(6) << avg_time << ", Result = " << last_result << "\n";
        }
        elapsed_time = 0.0;
    }

    if(proc_num == 0) {
        std::cout << "" << std::endl;
    }
    
    if (save_data && proc_num == 0) {
        if (!append) {
            vec_to_csv(time_results, filename + "_time");
            vec_to_csv(work_results, filename + "_work");
        } 
        else {
            std::ofstream time_out(filename + "_time.csv", std::ios::app);
            std::ofstream work_out(filename + "_work.csv", std::ios::app);

            for (int i = 0; i < max_size; ++i) {
                time_out << time_results[i] << "\n";
                work_out << work_results[i] << "\n";
            }
        }
    }

    return {time_results, work_results};
}

#endif // TIMER_HPP