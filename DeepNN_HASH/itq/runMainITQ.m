function [ map ] = runMainITQ( dataset, basedir, L  )
%MAINITQ this code runs ITQ 
    
    switch dataset
        case 'cifar-10'
            [ Xtrain, Xtest, gnd_inds] = config_cifar10(basedir);
        case 'mnist'
            [ Xtrain, Xtest, gnd_inds, Ytrain, Ytest] = config_mnist(basedir);
        otherwise
            error('do not know dataset');
    end
    gnd_inds = double(gnd_inds');
    [Btrain, Btest] = testITQ(Xtrain, Xtest, L, 'ITQ');
    map = KNNMap(Btrain, Btest, size(Btrain,1), gnd_inds) * 100;
    
end

