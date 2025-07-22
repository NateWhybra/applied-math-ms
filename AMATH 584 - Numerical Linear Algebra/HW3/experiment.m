[U, X] = qr(randn(80));
[V, X] = qr(randn(80));
S = diag(2 .^ (-1:-1:-80));
A = U*S*V';

[QC, RC] = cqr(A);
[QM, RM] = mqr(A);

r_vals_c = log(diag(RC));
r_vals_m = log(diag(RM));
j = (1:80)';

figure
scatter(j, r_vals_c);
hold on;
scatter(j, r_vals_m);
hold off;

xlabel('j');
ylabel('log(R_{jj})');
title('log(R_{jj}) vs. j');
legend('Classical QR (GS)', 'Modified QR (GS)');