function output = vectorized_f(z,funcname)
%% sigmoid
if strcmp (funcname,'sigmoid')
    output = 1 ./ (1+exp(-z)); 
end
%% tanh
if strcmp (funcname, 'tanh')
    output = tanh(z);
end