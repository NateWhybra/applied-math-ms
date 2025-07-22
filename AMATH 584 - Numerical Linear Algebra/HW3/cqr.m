% Compute the QR factorization using algorithm 7.1 (Classical QR Decomposition).
function [Q, R] = cqr(A)
    [m, n] = size(A);
  
    % Allocate memory for Q and R.
    Q = zeros(m, n);
    R = zeros(n, n);

    % Initialize the first column of Q by taking the first column vector of A and normalizing.
    Q(:, 1) = A(:, 1) / norm(A(:, 1), 2);

    % Compute the entries of Q and R.
    for j = 1:n
        v_j = A(:, j);
        
        for i = 1:j-1
            R(i, j) = Q(:, i)' * A(:, j);
            v_j = v_j - R(i, j) * Q(:, i);
        end

        R(j, j) = norm(v_j, 2);
        Q(:, j) = v_j / R(j, j);
    end
end