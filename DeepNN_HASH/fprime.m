function output = fprime(z,funcname)
%% sigmoid
    if strcmp(funcname,'sigmoid')
        a = vectorized_f(z,'sigmoid');
        output = a .* (1-a);
    end
%% tanh
    if strcmp(funcname,'tanh')
        a = vectorized_f(z,'tanh');
        output = 1 - a.^2;
    end
end