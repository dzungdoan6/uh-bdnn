function theta = initializeParameters(hiddenSize, visibleSize, inputData)

    cov_mat = (1/size(inputData,2)) * (inputData * inputData');
    [eigvec, eigval] = eig(cov_mat);
    eigvec = eigvec(:,end:-1:1); 
    W1 = eigvec(:,1:hiddenSize)';
    W2 = rand(visibleSize, hiddenSize); % not use in our network, just for general case
    b1 = zeros(hiddenSize, 1);
    b2 = zeros(visibleSize, 1); % not use in our network, just for general case
    theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

