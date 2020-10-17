function [ Xtrain, Xtest, gnd_inds, Ytrain, Ytest ] = config_cifar10(base_dir)
%CONFIG_CIFAR10_2 
    load([base_dir '/cifar-10/cifar_gist_320.mat']);
    load([base_dir '/cifar-10/groundtruth.mat']);
    Ytrain = double(Ytrain); Xtrain = double(Xtrain);
    Ytest = double(Ytest); Xtest = double(Xtest);
    
    
end

