% Compute the QR factorization using algorithm 8.1 (Modified QR Decomposition).
function [Q, R] = mqr(A)
    [m, n] = size(A);
  
    % Allocate memory for Q and R.
    Q = zeros(m, n);
    R = zeros(n, n);
    
    % This is basically the same as the first for loop setting v_i = a_i.
    V = A;

    % Compute the entries of Q and R.
    for i = 1:n
        R(i, i) = norm(V(:, i), 2);
        Q(:, i) = V(:, i) / R(i, i);
        
        for j = i+1:n
            R(i, j) = Q(:, i)' * V(:, j);
            V(:, j) = V(:, j) - R(i, j) * Q(:, i);
        end
    end
end