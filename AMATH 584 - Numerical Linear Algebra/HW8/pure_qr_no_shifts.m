function [B, Q] = pure_qr_no_shifts(A, stop)
    B = A;
    m = size(A, 1);
    % A matrix with 1's on the off-diagonals and 0's on the diagonals.
    % Used to conveniently grab the matrix entries I want to compute the
    % error.
    off_diagonal_mask = logical(ones(m, m) - eye(m));
    
    error = inf;
    while error > stop
        [Q, R] = qr(B);
        B = R * Q;
        % Here the error is the maximum of the off-diagonal entries.
        error = sum(max(abs(B(off_diagonal_mask))));  
    end
end