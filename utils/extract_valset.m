function [ Xval, val_gnd_inds ] = extract_valset( X, nval, ngnd )
    addpath(genpath('yael'));
    ntrain = size(X, 2);
    rand_inds = randperm(ntrain);
    Xval = X(:,rand_inds(1:nval));

    distab = yael_L2sqr(single(X), single(Xval));
    [~,val_gnd_inds] = yael_kmin(distab, ngnd);
    val_gnd_inds = double(val_gnd_inds');
    Xval = Xval';
end

