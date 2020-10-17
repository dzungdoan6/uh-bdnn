function [stack] = learn_all(Xtrain, Xval, val_gnd_inds, hiddenSize, ...
    lambda1, lambda2, lambda3, lambda4, iter_lbfgs, max_iter)
    %% parameters
    nl = 5; %number of layers (including input and output layer)
    trainData = Xtrain'; % each sample should be each column to be consistent with the paper
    inputSize = size(trainData,1);
    outputSize = inputSize;

    fprintf('Number of layers in NN = %d\n', nl);
    fprintf('\t %d --> %d --> %d --> %d --> %d\n', inputSize, hiddenSize(1), ...
        hiddenSize(2), hiddenSize(3), outputSize);
    fprintf('Number iteration of L-BFGS = %d\n', iter_lbfgs);
    fprintf('Maximum iteration of alternating optimization = %d\n', max_iter);
    %% initialize B using ITQ
    B = runITQ(trainData',hiddenSize(3))'; 

    %% initialize (W,c) using intialized B
    pre_training = true; % set true to initialize (W,c)

    stack = cell(nl-1,1);
    [stackout] = learnDeepNN(trainData, B, inputSize, hiddenSize,...
        lambda1, lambda2, lambda3, lambda4, pre_training, stack, 0,iter_lbfgs);
    stack = stackout;

    pre_training = false; % set false because (W,c) have been initialized already
    m = size(trainData,2);

    %% alternating learning B and (W,c) 
    map_best = realmin;
    for t = 1:max_iter
        fprintf('Iter %d\n', t);

        fprintf('\t learn B \n');
        [ H ] = feedForwardDeep(stack, trainData);
        Y = trainData - stack{nl-1}.b * ones(1,m);
        B = learn_B(Y', stack{nl-1}.w', B', H', lambda2)'; 

        fprintf('\t learn W \n');
        [stack, ~, ~, cost] = learnDeepNN(trainData, B, inputSize, hiddenSize,...
                                        lambda1, lambda2, lambda3, lambda4, ...
                                        pre_training, stack, t,iter_lbfgs);

        fprintf('\t cost function = %f\n', cost);
        
         %% Stopping criterion
         % We check mAP on validation set, if mAP on validation set is not
         % improved, we stop training
        Htrain = feedForwardDeep(stack, Xtrain')';
        Hval = feedForwardDeep(stack, Xval')';
        Btrain = zeros(size(Htrain));
        Btrain(Htrain >=0) = 1;
        Bval = zeros(size(Hval));
        Bval(Hval >= 0) = 1;
        map = KNNMap(Btrain, Bval, size(Btrain,1), val_gnd_inds);
        fprintf('\t mAP on validation set = %.3f\n', map*100);
        if map <= map_best
            stack = stack_best;
            break;
        else
            map_best = map;
            stack_best = stack;
        end
    end
    disp('\nFinish\n');
end
