function max_eig = power_iteration(A, stop)
    m = size(A, 1);
    v = rand(m, 1);
    v = v / norm(v, 2);
    
    error = inf;
    while error > stop
       w = A * v;
       v = w / norm(w, 2);
       max_eig = v.' * A * v;
       error = norm(A*v - max_eig*v, 2);
    end 
end