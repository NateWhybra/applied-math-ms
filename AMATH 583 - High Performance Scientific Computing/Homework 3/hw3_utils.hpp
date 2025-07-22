# ifndef HW3UTILS_HPP
# define HW3UTILS_HPP
# include <vector>

std::vector<std::vector<double>> random_matrix(int m, int n, double low=0.0, double high=1.0);
std::vector<double> random_vector(int n, double low=0.0, double high=1.0);
void mat_to_csv(const std::vector<std::vector<double>> &A, const std::string &name);
void vec_to_csv(const std::vector<double> &v, const std::string &name);

# endif