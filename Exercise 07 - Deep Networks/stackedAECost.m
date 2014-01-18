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


%[cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data);

% visibleSize: the number of input units (probably 784) 
% hiddenSize: the number of hidden units (probably 196) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

% for sparse auto encoder part
% --------: Forward Pass :--------
W1 = stack{1}.w;
W2 = stack{2}.w;
b1 = stack{1}.b;
b2 = stack{2}.b;

m = size(data, 2);
a1 = data;		% 784 x 10000
z2 = W1 * a1 + repmat(b1, 1, m);
a2 =  sigmoid(z2);
z3 = W2 * a2 + repmat(b2, 1, m);
a3 = sigmoid(z3);

% for the softmax part
z4 = softmaxTheta * a3;
a4 = exp(z4) ./ repmat(sum(exp(z4)), numClasses, 1);

% --------: Backward Pass :---------
delta4 = -(groundTruth - a4);
delta3 = (softmaxTheta' * delta4) .* sigmoidGrad(z3);

% back to sae part
delta2 =  (W2 * delta3) .* sigmoidGrad(z2);

W2grad = (delta3 * a2') * (1/m) + lambda * W2;
W1grad = (delta2 * a1') * (1/m) + lambda * W1;
b2grad = sum(delta3, 2) * (1/m);
b1grad = sum(delta2, 2) * (1/m);

stackgrad{2}.w = W2grad;
stackgrad{1}.w = W1grad;
stackgrad{2}.b = b2grad;
stackgrad{1}.b = b1grad;


% cost =  softmax cost only 
cost = -(1.0/m) * sum(sum(groundTruth .* log(a4))) + (lambda/2.0) * sum(sum(softmaxTheta .^2)) ... 	% soft max cost only
		+ (lambda/2) * ( sumsqr(W1) + sumsqr(W2) )		;											% W1/W2 regularization only



% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end

function grad = sigmoidGrad(x)
    e_x = exp(-x);
    grad = e_x ./ ((1 + e_x).^2); 
end

% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function ans = sumsqr(x)
	ans = sum(sum(x.^2));
end
