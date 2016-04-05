function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.


hypothesis_product = theta * data;
alpha = max(hypothesis_product);
disp(size(alpha));
disp(size(hypothesis_product));
hypothesis_product = hypothesis_product - repmat(alpha,10,1);

hypothesis = exp(1).^(hypothesis_product);
normalize = sum(hypothesis);
hypothesis_norm = bsxfun(@rdivide,hypothesis,normalize);

for i=1:size(data, 2)
    [val idx] = max(hypothesis_norm(:,i));
    pred(i) = idx;
end

% ---------------------------------------------------------------------

end

