function mAP = runHash( dataset, basedir, L)
%RUNHASH run demos
%INPUT
%   dataset: name of dataset, it should be cifar-10 or mnist
%   basedir: folder contains dataset
%   L: number of bits to encoder each vector
%OUTPUT
%   mAP: mean average precision
    %% Setup parameters and dataset
    max_iter = 20; % Maximum iteration of alternating optimization B and W
    iter_lbfgs = 400; % number iteration of L-BFGS for learning weights W
    lambda1 = 10^-5;
    lambda2 = 5*10^-2;
    lambda3 = 10^-2;
    lambda4 = 0.5*10^-6;

    % Configure number of layers 
    switch L
        case 8
            hiddenSize(1) = 90;
            hiddenSize(2) = 20;
        case 16
            hiddenSize(1) = 90;
            hiddenSize(2) = 30;
        case 24
            hiddenSize(1) = 100;
            hiddenSize(2) = 40;
        case 32
            hiddenSize(1) = 120;
            hiddenSize(2) = 50;
        otherwise
            error('please specify L = 8, 16, 24 or 32');
    end
    hiddenSize(3) = L;

    % Load dataset
    switch dataset
        case 'cifar-10'
            [ Xtrain, Xtest, gnd_inds] = config_cifar10(basedir);
        case 'mnist'
            [ Xtrain, Xtest, gnd_inds, Ytrain, Ytest] = config_mnist(basedir);
        otherwise
            error('do not know dataset');
    end
    
    % Preprocess data
    [Xtrain, Xtest] = rescale_0_1(Xtrain, Xtest);
    nval = 1000; % size of validation set
    nval_gnd = size(gnd_inds, 1); % number of groundtruth in validation set should be same as test set
    [ Xval, val_gnd_inds ] = extract_valset(Xtrain', nval, nval_gnd); % extract validation set
    gnd_inds = double(gnd_inds');
    
    % Display data info
    fprintf('Dataset = %s\n', dataset);
    fprintf('\t Number of train = %d\n', size(Xtrain, 1));
    fprintf('\t Number of test = %d\n', size(Xtest, 1));
    fprintf('\t Number of validation = %d\n', size(Xval, 1));

    %% Train deep neural networks
    tic;
    stack = learn_all(Xtrain, Xval, val_gnd_inds, hiddenSize, ...
        lambda1, lambda2, lambda3, lambda4, iter_lbfgs, max_iter);
    fprintf('Train in %.3fs\n', toc);

    %% Evaluation
    Htrain = feedForwardDeep(stack, Xtrain')';
    Htest = feedForwardDeep(stack, Xtest')';
    Btrain = zeros(size(Htrain));
    Btrain(Htrain >= 0) = 1;
    Btest = zeros(size(Htest));
    Btest(Htest >= 0) = 1;

    mAP = KNNMap(Btrain,Btest,size(Btrain,1),gnd_inds) * 100;
    
end

