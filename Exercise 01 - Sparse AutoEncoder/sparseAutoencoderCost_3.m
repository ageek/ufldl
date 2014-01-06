function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

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
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

%data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
% for sparse auto-encoder output=input

% size(data, 1) % 64
% size(W1)   % 25 64
% size(W2)   % 64 25
% size(b1)   % 25  1
% size(b2)   % 64  1
m = size(data,2);

% calculate a1 : size(data): 64x10000
% input: 64 input units
a1 = data;		% 64 x 10000

% size(W1) : 25 x 64
% size(a2) : 25 x 10000
% size(z2): 25 x 10000
% include intercept term, to be added to each 25 columns 
z2 = W1 * a1 + repmat(b1, 1, m);
a2 =  sigmoid(z2);

% calculate average activation: rho_hat : 1x25
% sum across all examples m
rho_hat = (1/m) * sum(a2,2);	


%h_theta : predicted output, should be 10000 x64   = 10000x25   X 25x64
% size(a2) : 25 x 10000
% size(W2): 64x25
% size(z3) : 64 x 10000
% include intercept term, to be added to each 64 columns 
z3 = W2 * a2 + repmat(b2, 1, m);
a3 = sigmoid(z3);


diff = a3 - data;
J_unreg = (1/2*m) * sumsqr(diff);
regularization = (lambda/2) * (sumsqr(W1) + sumsqr(W2) );
sparse_penalty = kl(sparsityParam, rho_hat);
cost = J_unreg + beta * sparse_penalty + regularization;

	
% calculate KL-divergence term to be added to delta2
rho=sparsityParam;
kl_delta = beta * ( -(rho ./ rho_hat) + (1-rho) ./ (1-rho_hat))	;

% size(delta3): 64x10000
delta3 = diff .* sigmoidGradient(z3);

% size(delta2) : 25x10000
delta2 =  ((W2' * delta3) + repmat(kl_delta, 1, m)) .* sigmoidGradient(z2);

% Backpropogation
% f'(z) = a * (1-a)
y = data;

% for i=1:m,
%  delta3 = -(y(:,i) - a3(:,i)) .* sigmoidGradient(z3(:,i)); 
%   delta2 = W2'*delta3 .* sigmoidGradient(z2(:,i));
 
%   W2grad = W2grad + delta3*a2(:,i)' * (1/m);
%   W1grad = W1grad + delta2*a1(:,i)' * (1/m); 
%   b2grad = b2grad + delta3 * (1/m);
%   b1grad = b1grad + delta2 * (1/m);
% end;



% Actual Grad is DELTA * (1/m)
W1grad = (delta2 * a1') * (1/m) + lambda * W1;
W2grad = (delta3 * a2') * (1/m) + lambda * W2;

b2grad = sum(delta3, 2) * (1/m);
b1grad = sum(delta2, 2) * (1/m);
%W2grad = W2grad + lambda * W2;
%W1grad = W1grad + lambda * W1;


size(W1grad);
size(W2grad);
size(b1grad);
size(b2grad);


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

function g = sigmoidGradient(z)
	g = zeros(size(z));
	g = sigmoid(z) .* ( 1- sigmoid(z));
end

function div = kl(r, rh)
  div = sum(r .* log(r ./ rh) + (1-r) .* log( (1-r) ./ (1-rh)));
end
