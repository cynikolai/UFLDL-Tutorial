function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));

%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

W1 = stack{1}.w;
b1 = stack{1}.b;
W2 = stack{2}.w;
b2 = stack{2}.b;

num = size(data,2);

z2 = W1 * data + repmat(b1,1,num);
a2 = arrayfun(@sigmoid,z2);

z3 = W2 * a2 + repmat(b2,1,num);
a3 = arrayfun(@sigmoid,z3);

hypothesis_product = softmaxTheta * a3;
alpha = max(hypothesis_product);
hypothesis_product = hypothesis_product - repmat(alpha,10,1);
hypothesis = exp(1).^(hypothesis_product);
normalize = sum(hypothesis);
hypothesis_norm = bsxfun(@rdivide,hypothesis,normalize);

cost = (-1/m) * sum(sum(groundTruth .* log(hypothesis_norm) + (lambda / 2) * theta .^2));
softmaxThetaGrad = (-1/m) * (groundTruth - hypothesis_norm) * transpose(a3) + lambda * softmaxTheta;

%delta
delta_3 = -(transpose(softmaxTheta) * (groundTruth - hypothesis_norm)) .* a3 .* (1 -a3);
delta_2 = stack{2}.w * delta{3} .* a2 .* (1 -a2) ;
delta_1 = stack{1}.w * delta{2} .* a1 .* (1 -a1) ;

W1grad = delta_2 * transpose(data)/num+ lambda * W1; 
W2grad = delta_3 * transpose(a2)/num+ lambda * W2 ;
b1grad = sum(delta_2,2)/num; 
b2grad = sum(delta_3,2)/num;

stackgrad{1}.w = W1grad;
stackgrad{1}.b = b1grad;
stackgrad{2}.w = W2grad;
stackgrad{2}.b = b2grad;


% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function sigm = sigderiv(x)
    sigm = sigmoid(x) .* (1-sigmoid(x));
end 
