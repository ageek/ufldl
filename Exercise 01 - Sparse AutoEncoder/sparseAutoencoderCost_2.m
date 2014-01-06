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
y = data';

% calculate a1 : size(data): 64x10000
% input: 64 input units
a1 = data';		% 10000x 64

% size(W1) : 25 x 64
% size(a2) : 10000x64   X 64x25  = 10000x25
% include intercept term, to be added to each 25 columns 
a2 =  sigmoid( bsxfun(@plus, b1',(a1*W1')) );


%h_theta : predicted output, should be 10000 x64   = 10000x25   X 25x64
% size(a2) : 10000x25 
% size(W2): 64x25
% include intercept term, to be added to each 64 columns 
h_theta = sigmoid( bsxfun(@plus, b2', (a2 * W2')) );

m = size(data,2);
J_unreg = (1/m) * sumsqr(h_theta - data');

regularization = (lambda/2) * ( sumsqr(W1) + sumsqr(W2) );
J = J_unreg + regularization;

cost = J;

% calculate average activation: rho_hat
a2_sum = zeros(1,25);
for i=1:m;
	a1 = data(:,i)';
	a2 =  sigmoid( bsxfun(@plus, b1',(a1*W1')) );
	a2_sum = a2_sum + a2;
end;


rho_hat = (1/m) * sum(a2_sum);	
rho = sparsityParam;
	
% calculate KL-divergence term to be added to delta2
kl_term = beta * ( -(rho/rho_hat) + (1-rho)/(1-rho_hat))	;

% for accumulating deltas in each iteration
% for W1 and W2
DELTA_W1 = zeros(size(W1));
DELTA_W2 = zeros(size(W2));
% for b1 and b2
DELTA_b1 = zeros(size(b1));
DELTA_b2 = zeros(size(b2));

for i=1:m;
	%fprintf('Iteration %d\n',i);
	% pick one example from X's 10000 entries
	% a1:  1x64
	% data: 64x10000
	a1 = data(:,i)';
	
	% include intercept term, to be added to each of the 25 columns 
	% size(W1) : 25 x 64
	% size(b1): 25 x 1
	% size(a2): 1x25
	a2 =  sigmoid( bsxfun(@plus, b1',(a1*W1')) );
	
	% include intercept term, to be added to each of the 64 columns 
	% size(W2): 64x25
	% size(b2): 64 x 1
	% h_theta = a3   % size: 1x64
	a3 = sigmoid( bsxfun(@plus, b2', (a2 * W2')) );
	
	% calculate delta3 : 1 x 64
	% a3 = a1 		%sparse auto-encoder
	delta_3 = a3 - data(:,i)';
	
	% calculate deltas
	% size(delta3 X W2) : 1x64 X 64x25 = 1x25
	delta3_x_W2 = (delta_3 * W2) ;
	
	% also include KL-divergence term
	delta3_x_W2 = bsxfun(@plus, delta3_x_W2, kl_term);
	
	%size(delta_2) = 1x25
	delta_2 = delta3_x_W2 .* sigmoidGradient( bsxfun(@plus, b1',(a1*W1')) );

	
	% size(DELTA_W2): size(delta_3') * a2 = 64 x25
	DELTA_W2 = DELTA_W2 + delta_3' * a2;
	% size(DELTA_W1): 25 x 64
	DELTA_W1 = DELTA_W1 + delta_2' * a1;
	
	% size(DELTA_b2): 64 x 1
	DELTA_b2 = DELTA_b2 + delta_3';
	% size(DELTA_b1): 25 x 1
	DELTA_b1 = DELTA_b1 + delta_2';
	
end;


% Actual Grad is DELTA * (1/m)
W1grad = DELTA_W1 * ( 1/m);
W2grad = DELTA_W2 * ( 1/m);

b2grad = DELTA_b2 * ( 1/m);
b1grad = DELTA_b1 * ( 1/m);

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
