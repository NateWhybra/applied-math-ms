% Store our epsilon values of interest in a list.
eps = [1, 10^(-2), 10^(-4), 10^(-6), 10^(-8)];

% Make an empty matrix to store our results.
results = zeros(5, 3);

% Repeat this procedure for each epsilon.
for i=1:5
    % Create our matrix of interest A.
    A = [1 1 1; eps(i) 0 0; 0 eps(i) 0; 0 0 eps(i)];

    % Compute each version of the QR factorization.
    [Q1, R1] = cqr(A);
    [Q2, R2] = mqr(A);
    [Q3, R3] = house_qr(A);

    % Get the identity matrix.
    I = eye(3);

    % Store the norm errors as a row in our results matrix.
    results(i, :) = [norm(Q1' * Q1 - I, 2), norm(Q2' * Q2 - I, 2), norm(Q3' * Q3 - I, 2)];
end

% Make plots to summarize our findings.
figure;
loglog(eps, results, '-o');
xlabel('Epsilon');
ylabel('Error')
title('Error vs. Espilon')
legend('Classical QR', 'Modified QR', 'Householder QR')