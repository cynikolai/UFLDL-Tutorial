function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------

% visibleSize: the number of input units (probably 64)
% hiddenSize: the number of hidden units (probably 25)
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
                                                                      %                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.

% The input theta is a vector (because minFunc expects the parameters to be a vector).
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
% follows the notation convention of the lecture notes.

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values).
% Here, we initialize them to zeros.
cost = 0;
W1grad = zeros(size(W1));
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b)
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code
% in Section 2.2
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
%
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2.
%
% Feedforward

num = size(data,2);

z2 = W1 * data + repmat(b1,1,num);
a2 = arrayfun(@sigmoid,z2);

rho = sum(a2,2)/num;
rho_cost = beta * sum(arrayfun(@sparsity,rho));

z3 = W2 * a2 + repmat(b2,1,num);
a3 = z3

% Cost

cost = .5 * sum(sum((a3 - data).^2))/ num;
cost = cost  + (lambda/2)*(sum(sum(W1.^2)) + sum(sum(W2.^2)));
cost = cost + rho_cost;

% Delta

% Output Layer
delta_3 = -(data - a3);

% Hidden Layer
delta_2 = ((transpose(W2) * delta_3) + repmat(beta * arrayfun(@sparsity_deriv,rho),1,num)) .* arrayfun(@sigderiv,z2);

% partial derivatives
W1grad = delta_2 * transpose(data)/num+ lambda * W1;
W2grad = delta_3 * transpose(a2)/num+ lambda * W2 ;
b1grad = sum(delta_2,2)/num;
b2grad = sum(delta_3,2)/num;


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
                                                                      % column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)).

function sigm = sigmoid(x)

sigm = 1 ./ (1 + exp(-x));
end

function deriv = sigderiv(x)

deriv = sigmoid(x) * (1 - sigmoid(x));
end

function rho_cost = sparsity(x)
sparsityParam = 0.01;
rho_cost = sparsityParam * log(sparsityParam/x) + (1 - sparsityParam) * log((1-sparsityParam)/(1-x));
end

function rho_deriv = sparsity_deriv(x)
sparsityParam = 0.01;
rho_deriv = (-(sparsityParam/x) + (1-sparsityParam)/(1-x));
end

