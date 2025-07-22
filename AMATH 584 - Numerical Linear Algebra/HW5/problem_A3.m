% Make the matrix A as defined in the problem.
n = 100;
A = -1 * tril(ones(n), -1) + eye(n);
A(:, end) = 1; 

% Make the random vector.
x = randn(100, 1);

% Calculate b.
b = A * x;

% Part (a).
k = cond(A);

% Part (b).
x_ge = A \ b;
norm_error_1 = norm(x - x_ge, 2);

% Part (c).
[Q,R] = qr(A, 0); 
x_qr = R \ (Q' * b);
norm_error_2 = norm(x - x_qr, 2);

% Part (d).
% P*A*Q = L*U...
[L, U, P, Q] = gecp(A);
x_gecp = Q * inv(U) * inv(L) * P * b;
norm_error_3 = norm(x - x_gecp, 2);



