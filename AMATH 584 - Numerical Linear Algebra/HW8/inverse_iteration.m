function eigen_vector = inverse_iteration(A, lambda, stop)
    m = size(A, 1);
    v = rand(m, 1);
    v = v / norm(v, 2);

    error = inf;
    I = eye(m);
    while error > stop
        w = (A - lambda * I) \ v;
        v = w / norm(w, 2);
        error = norm(A * v - lambda * v, 2);
    end

    eigen_vector = v;
end