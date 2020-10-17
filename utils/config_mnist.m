function [ Xtrain, Xtest, gnd_inds, Ytrain, Ytest] = config_mnist( basedir)
    load([basedir '/mnist/mnist_train.mat']);
    load([basedir '/mnist/mnist_test.mat']);
    load([basedir '/mnist/groundtruth.mat']);

    % each sample locates in each row to make consistent as cifar-10
    Xtrain = Xtrain'; 
    Xtest = Xtest'; 

   
  
end

