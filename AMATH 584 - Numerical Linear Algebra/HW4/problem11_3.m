% Make our dimensions.
m = 50;
n = 12;

% Make our Vandermonde matrix.
t = linspace(0, 1, 50)';
A = fliplr(vander(t));
A = A(:, 1:n);

% Make fake data for us to fit.
b = cos(4 * t);

% Compute least squares solutions to Ax=b in different ways.
% Part (a):
x_a = (A' *A) \ (A' * b);

% Part (d) (Note: The second input in qr is to tell MATLAB to give me the reduced QR):
[Q, R] = qr(A, 0);
x_d = inv(R) * (Q' * b);

% Part (e):
x_e = A \ b;

% Part (f)  (Note: The second input in svd is to tell MATLAB to give me the reduced SVD):
[U, S, V] = svd(A, 0);
w = inv(S) * U' * b;
x_f = V * w; 

% Part (g):
solutions = [x_a, x_d, x_e, x_f];




