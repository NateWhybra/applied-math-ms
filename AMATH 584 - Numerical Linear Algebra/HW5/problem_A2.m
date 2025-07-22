x0 = 1.2;
f_prime_exact = cos(x0);
h = 10.^(0:-1:-16);
f_prime_approx = (sin(x0 + h) - sin(x0)) ./ h;
better_approx = (2 * cos(x0 + h/2) .* sin(h/2)) ./ h; 
error_1 = abs(f_prime_exact - f_prime_approx);
error_2 = abs(f_prime_exact - better_approx);

figure;
loglog(h, error_1);
xlabel("h")
ylabel("Error (b)")
title("Error (b) vs. h")

figure;
loglog(h, error_2);
xlabel("h")
ylabel('Error (d)')
title('Error (d) vs. h')