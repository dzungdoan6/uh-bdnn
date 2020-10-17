function [activation] = feedForwardDeepNN(theta, hiddenSize, visibleSize, data)

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

z2 = bsxfun(@plus, W1 * data, b1);
a2 = vectorized_f(z2,'sigmoid');
activation = a2;


end


