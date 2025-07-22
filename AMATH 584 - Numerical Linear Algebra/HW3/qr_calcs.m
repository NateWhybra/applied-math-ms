A = [1 0; 0 1; 1 0];
P1 = A * inv(A' * A) * A'
v = [1; 2; 3];
x1 = P1 * v;

B = [1 2; 0 1; 1 0];
P2 = B * inv(B' * B) * B';
x2 = P2 * v

Q1 = [1/sqrt(2) 0; 0 1; 1/sqrt(2) 0];
R1 = [sqrt(2) 0; 0 1];
A1 = Q1 * R1;

[Q3, R3] = mqr(B);

Q3
R3




