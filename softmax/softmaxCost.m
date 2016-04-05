function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.



hypothesis_product = theta * data;
alpha = max(hypothesis_product);
disp(size(alpha));
disp(size(hypothesis_product));
hypothesis_product = hypothesis_product - repmat(alpha,10,1);
hypothesis = exp(1).^(hypothesis_product);
normalize = sum(hypothesis);
hypothesis_norm = bsxfun(@rdivide,hypothesis,normalize);

for i=1:60000
    cost = cost - 1/60000 * log(hypothesis_norm(labels(i),i)) + lambda/2 * sum(sum(theta))^2;
end


for j=1:10
    thetagrad(j,:) = lambda/2 * theta(j,:);
end
for i=1:60000
    for j=1:10
        ind = 0;
        if(labels(i) == j)
           ind = 1;
        end
        thetagrad(j,:) = thetagrad(j,:) - (1/60000 * transpose(data(:,i)))*(ind - hypothesis_norm(j,i));
    end
    disp(i);
end



% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

