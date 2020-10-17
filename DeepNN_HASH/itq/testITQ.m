function [ Ytrain, Ytest ] = testITQ( Xtrain, Xtest, bit, method )
%TESTITQ runs and evaluates ITQ

num_training = size(Xtrain,1);

% generate training ans test split and the data matrix
XX = [Xtrain; Xtest];
% center the data, VERY IMPORTANT
sampleMean = mean(XX,1);
XX = (XX - repmat(sampleMean,size(XX,1),1));

%several state of art methods
switch(method)
    
    % ITQ method proposed in our CVPR11 paper
    case 'ITQ'
        % PCA
        [pc, l] = eigs(cov(XX(1:num_training,:)),bit);
        XX = XX * pc;
        % ITQ
        [Y, R] = ITQ(XX(1:num_training,:),50);
        XX = XX*R;
        Y = zeros(size(XX));
        Y(XX>=0) = 1;
%         Y = compactbit(Y>0);
    % RR method proposed in our CVPR11 paper
    case 'RR'
        % PCA
        [pc, l] = eigs(cov(XX(1:num_training,:)), bit);
        XX = XX * pc;
        % RR
        R = randn(size(XX,2),bit);
        [U S V] = svd(R);
        XX = XX*U(:,1:bit);
        Y = compactbit(XX>0);
   % SKLSH
   % M. Raginsky, S. Lazebnik. Locality Sensitive Binary Codes from
   % Shift-Invariant Kernels. NIPS 2009.
    case 'SKLSH' 
        RFparam.gamma = 1;
        RFparam.D = D;
        RFparam.M = bit;
        RFparam = RF_train(RFparam);
        B1 = RF_compress(XX(1:num_training,:), RFparam);
        B2 = RF_compress(XX(num_training+1:end,:), RFparam);
        Y = [B1;B2];
    % Locality sensitive hashing (LSH)
     case 'LSH'
        XX = XX * randn(size(XX,2),bit);
        Y = zeros(size(XX));
        Y(XX>=0)=1;
        Y = compactbit(Y);
end
% compute Hamming metric and compute recall precision
Ytrain = Y(1:size(Xtrain,1),:);
Ytest = Y(size(Xtrain,1)+1:end,:);
end

