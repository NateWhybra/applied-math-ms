% Construct the matrix A.
m = 10;
v = ones(10, 1);
A = diag(2 * v, 0) + diag(-1 * v(1:m-1), -1) + diag(-1 * v(1:m-1), 1);

% Computing eigen-values to check my work.
lambs = eigs(A);

% Part (b).
max_eig = power_iteration(A, 1e-6);

% Part (c).
D = pure_qr_no_shifts(A, 1e-12);

% Part(d).
fifth_eig = D(5, 5);
eigen_vector_5 = inverse_iteration(A, fifth_eig, 1e-8);
error = norm(A * eigen_vector_5 - fifth_eig * eigen_vector_5, 2);
