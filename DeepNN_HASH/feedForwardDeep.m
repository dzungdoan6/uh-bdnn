function [ H ] = feedForwardDeep(stack, data)

%% Perform forward pass to compute activation a2, a3, a4, a5 for layer L2,
%% L3, L4, L5  
a{1} = data;
z{1} = [];
nl = numel(stack) + 1; %number of layers
for d = 1:numel(stack)
    z{d+1} = bsxfun(@plus, stack{d}.w * a{d}, stack{d}.b);
    if d+1 <= 3 %sigmoid activation function
        a{d+1} = vectorized_f(z{d+1},'sigmoid');
    else 
        if d+1 == 4 
%             a{d+1} = vectorized_f(z{d+1},'tanh'); %tanh activation function
            a{d+1} = z{d+1};
        else %linear (identity activation function)
            a{d+1} = z{d+1};
        end
    end
end
H = a{nl-1};
