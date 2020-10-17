function [ Xtrain, Xtest ] = rescale_0_1( Xtrain, Xtest )
%RESCALE_0_1 normalizes data to [0;1]
max_dims = max(Xtrain,[],1); 
min_dims = min(Xtrain,[],1); 
range_dims = max(max(max_dims-min_dims+eps)); 
Xtrain = bsxfun(@minus,Xtrain,min_dims); 
Xtrain = bsxfun(@rdivide,Xtrain,range_dims);
Xtest = bsxfun(@minus,Xtest,min_dims); 
Xtest = bsxfun(@rdivide,Xtest,range_dims);



end

