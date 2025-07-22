% Compute the reduced QR factorization of a matrix A using Householder Triangularization.
function [Q, R] = house_qr(A)
    % Get matrix dimensions.
    [m, n] = size(A);
    
    % Define a function for quickly getting standard basis vectors.
    % e(k, m) is a m x 1 column vector with a 1 in the kth row.
    e = @(k, m) [zeros(k-1,1); 1; zeros(m-k, 1)];

    % Make cell array for storing v_{k}'s.
    v_storage = cell(n, 1);

    % Algorithm (10.1).
    % Compute R and store it in A.
    for k = 1:n
        x = A(k:m, k);
        v = sign(x(1)) * norm(x, 2) * e(1, m-k+1) + x;
        v = v / norm(v, 2);
        A(k:m, k:n) = A(k:m, k:n) - 2 * v * (v' * A(k:m, k:n));
        v_storage{k} = v;
    end

    % Return R as a copy of A.
    R = A;
    % Make R a n x n matrix so that this is the reduced QR.
    R = R(1:n, :);

    % Make an empty matrix for Q.
    Q = zeros(m, n);

    % For each standard basis vector e_{i}, use Algorithm 10.3 to compute
    % Qe_{i} to get q_{i}.
    for i = 1:n
        q = e(i, m);
        for k = n:-1:1
            q(k:m) = q(k:m) - 2 * v_storage{k} * (v_storage{k}' * q(k:m)); 
        end
        Q(:, i) = q;
    end
end