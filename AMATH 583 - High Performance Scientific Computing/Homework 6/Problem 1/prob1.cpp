# include <iostream>
# include <vector>
# include <thread>
# include <cmath>
# include <stdexcept>
# include "timer.hpp"
# include "arc_length_c_threads.hpp"
# include "utils.hpp"
using namespace std;


int main() {
    // Get data for strong scaling (c).
    std::pair<std::vector<double>, std::vector<double>> res1 = time_work(1, 5, "problem_1_scaling", prepare_data_A, split_calc, true, false);

    // Get data for numerical errors (d).
    std::pair<std::vector<double>, std::vector<double>> res2 = time_work(1, 6, "problem_1_numerical", prepare_data_B, split_calc, false, false);
    
    double true_val = log(6.0) + (35.0 / 8.0);
    vector<double> exp_vals = res2.second;
    vector<double> errors(exp_vals.size());
    for(int i = 0; i < exp_vals.size(); i++) {
        errors[i] = log(abs(exp_vals[i] - true_val));
    }
    // Save numerical errors.
    vec_to_csv(errors, "problem_1_numerical_errors");
}





