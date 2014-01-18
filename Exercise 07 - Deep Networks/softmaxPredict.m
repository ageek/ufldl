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

% sae part
W1 = stack{1}.w;
W2 = stack{2}.w;
b1 = stack{1}.b;
b2 = stack{2}.b;
m = size(data, 2);
a1 = data;		
z2 = W1 * a1 + repmat(b1, 1, m);
a2 =  sigmoid(z2);
z3 = W2 * a2 + repmat(b2, 1, m);
a3 = sigmoid(z3);

% softmax part
z4 = softmaxTheta * a3;
a4 = exp(z4) ./ repmat(sum(exp(z4)), numClasses, 1);

% prediction 
[p,pred] = max(a4, [], 1);
% ---------------------------------------------------------------------

end

