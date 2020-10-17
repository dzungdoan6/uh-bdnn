function [ cost, grad, J1, J2, J3, J4, J5 ] = deepNNCost(theta, inputSize, hiddenSize, netconfig, ...
    lambda1, lambda2, lambda3, lambda4, B, data)
                                         
%% Extract stack 
% Extract out the "stack"
stack = params2stack(theta(1:end), netconfig);

stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this
m = size(data, 2); %number of data


%% Perform forward pass to compute activation a2, a3, a4, a5 for layer L2, L3, L4, L5  

a{1} = data;
X = data;
z{1} = [];
nl = numel(stack) + 1; %number of layers
for d = 1:numel(stack)
    z{d+1} = bsxfun(@plus, stack{d}.w * a{d}, stack{d}.b);
    if d+1 <= 3 %sigmoid activation function
        a{d+1} = vectorized_f(z{d+1},'sigmoid');
    else 
        if d+1 == 4
%             a{d+1} = vectorized_f(z{d+1},'tanh'); %tanh
            a{d+1} = z{d+1};
        else %linear (identity activation function)
            a{d+1} = z{d+1};
        end
    end
end

%% compute J (cost function)
J1 = ( 1/(2*m) ) * mnorm(X - stack{nl-1}.w*B - stack{nl-1}.b*ones(1,m));
J2 = 0;
for l = 1:nl-1
    J2 = J2 + mnorm(stack{l}.w);
end
J2 = (lambda1/2)*J2;
J3 = ( lambda2/(2*m) )* mnorm(a{nl-1}-B);
tmp = a{nl-1}*a{nl-1}';
J4 = (lambda3/2) * mnorm( (1/m)*tmp - eye(size(tmp,1)) );
J5 = (lambda4/(2*m)) * mnorm (a{nl-1}*ones(m,1));
cost = J1 + J2 + J3 + J4 + J5;
%% compute gradient of J w.r.t. stack{nl-1}.w
stackgrad{nl-1}.w = (-1/m) * (X - stack{nl-1}.w * B - stack{nl-1}.b*ones(1,m)) * B' + lambda1*stack{nl-1}.w;
stackgrad{nl-1}.b = (-1/m) * ( (X-stack{nl-1}.w*B)*ones(m,1) - m*stack{nl-1}.b);
%% compute gradient of J w.r.t. others w
 delta{nl-1} = (lambda2/m) * (a{nl-1} - B) + (2*lambda3/m)*( (1/m)*tmp - eye(size(tmp,1)) ) * a{nl-1} ...
     + (lambda4/m)*(a{nl-1}*ones(m,1)*ones(1,m)); %if at layer (nl-1) using identity activation f(z{nl-1}) = z{nl-1} --> f'(z{nl-1}) = 1

%delta{nl-1} = delta{nl-1} .* fprime(z{nl-1},'tanh'); %tanh

for l = nl-2:-1:2
    delta{l} = (stack{l}.w' * delta{l+1}) .* fprime(z{l},'sigmoid');
end

for l = (nl-2):-1:1
    stackgrad{l}.w = delta{l+1} * a{l}' + lambda1 * stack{l}.w;
    stackgrad{l}.b = sum(delta{l+1},2); % = delta{l+1} * ones(m,1)
end
% -------------------------------------------------------------------------

%% Roll gradient vector
[grad, netconfig] = stack2params(stackgrad);
end

