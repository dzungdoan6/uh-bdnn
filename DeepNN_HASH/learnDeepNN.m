%% deepAE_Hash Exercise
function [stackout,deepNNOptTheta,netconfig, cost] = learnDeepNN(trainData,B,inputSize,hiddenSize,...
    lambda1, lambda2, lambda3, lambda4, pre_training, stack, t, iter_lbfgs)

outputSize = inputSize;

if pre_training
    %% Initialize W1
    sae1OptTheta = initializeParameters(hiddenSize(1), inputSize, trainData);

    
    %% Initialize W2
    [sae1Features] = feedForwardDeepNN(sae1OptTheta, hiddenSize(1), inputSize, trainData);
    sae2OptTheta = initializeParameters(hiddenSize(2), hiddenSize(1), sae1Features);
    
    %% Initialize W3
    [sae2Features] = feedForwardDeepNN(sae2OptTheta, hiddenSize(2), hiddenSize(1), sae1Features);
    sae3OptTheta = initializeParameters(hiddenSize(3), hiddenSize(2), sae2Features);
   

    %% Initialize W4
    W4_1 = eye(outputSize, hiddenSize(3));
    W4_2 = rand(hiddenSize(3), outputSize); % not use in our network, just for general case
    b4_1 = zeros(outputSize, 1);
    b4_2 = zeros(hiddenSize(3), 1);  % not use in our network, just for general case
    sae4OptTheta = [W4_1(:) ; W4_2(:) ; b4_1(:) ; b4_2(:)];
    
    
    %% Stack all initialized (W,c)
    stack = cell(4,1);
    stack{1}.w = reshape(sae1OptTheta(1:hiddenSize(1)*inputSize), hiddenSize(1), inputSize);
    stack{1}.b = sae1OptTheta(2*hiddenSize(1)*inputSize+1:2*hiddenSize(1)*inputSize+hiddenSize(1));
    stack{2}.w = reshape(sae2OptTheta(1:hiddenSize(2)*hiddenSize(1)), hiddenSize(2), hiddenSize(1));
    stack{2}.b = sae2OptTheta(2*hiddenSize(2)*hiddenSize(1)+1:2*hiddenSize(2)*hiddenSize(1)+hiddenSize(2));
    stack{3}.w = reshape(sae3OptTheta(1:hiddenSize(3)*hiddenSize(2)), hiddenSize(3), hiddenSize(2));
    stack{3}.b = sae3OptTheta(2*hiddenSize(3)*hiddenSize(2)+1:2*hiddenSize(3)*hiddenSize(2)+hiddenSize(3));
    stack{4}.w = reshape(sae4OptTheta(1:outputSize*hiddenSize(3)), outputSize, hiddenSize(3));
    stack{4}.b = sae4OptTheta(2*outputSize*hiddenSize(3)+1:2*outputSize*hiddenSize(3)+outputSize);
end
%%======================================================================
%% Train Deep NN by back propagation
[stackparams, netconfig] = stack2params(stack); % stack all parametter to a vector
deepNNTheta = stackparams;
options.Method = 'lbfgs';
options.maxIter = iter_lbfgs;	  % 400 Maximum number of iterations of L-BFGS to run 
options.display = 'off';

%% run L-BFGS
[deepNNOptTheta, cost] = minFunc( @(p) deepNNCost(p, inputSize, hiddenSize, netconfig, ...
                                   lambda1, lambda2, lambda3, lambda4, B, trainData), ...
                              deepNNTheta, options);

stackout = params2stack(deepNNOptTheta(1:end), netconfig);
