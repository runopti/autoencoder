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
%disp('OK');
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
m = size(data, 2);

deltaW1 = W1grad;
deltaW2 = W2grad;
deltab1 = b1grad;
deltab2 = b2grad;


% for i = 1:m
%     input = data(:,i);
%     %compute the current cost
%     
%     %perform a feedworward pass up to output layer
%     hidden = sigmoid(W1*input+b1);
%     output = sigmoid(W2*hidden+b2);
%     
%     %compute the current cost
%     cost_i = 1/2*norm(input-output);
%     cost = cost + cost_i;
%     
%     %start backpropagation
%     %set the error term for the output layer
%     delta3 = -(input-output).*(output.*(1-output));
%     delta2 = (W2.'*delta3).*(hidden.*(1-hidden));
%     delta1 = (W1.'*delta2).*(input.*(1-input));
%     
%     partialW1J = delta2*(input.');
%     partialb1J = delta2;
%     partialW2J = delta3*(hidden.');
%     partialb2J = delta3;
%     
%     
%     deltaW1 = deltaW1 + partialW1J;
%     deltab1 = deltab1 + partialb1J;
%     deltaW2 = deltaW2 + partialW2J;
%     deltab2 = deltab2 + partialb2J;
%     
% end

%forward
input = data;
hidden = sigmoid(W1*input+repmat(b1,1,m));
output = sigmoid(W2*hidden+repmat(b2,1,m));

disp('OK1');

rho_hat = sum(hidden, 2)/m;
% rho_hat_rep = repmat(rho_hat, m);
cost = sum(sum((output-input).^2));

% disp('OK2');
%backpropagation
delta3 = -(input-output).*(output.*(1-output));
delta2 = (W2.'*delta3 + beta*repmat(-sparsityParam./rho_hat + (1-sparsityParam)./(1-rho_hat),1,m)).*(hidden.*(1-hidden));
delta1 = (W1.'*delta2).*(input.*(1-input));
% disp('OK3');
partialW1J = delta2*(input.');
partialb1J = sum(delta2,2);
partialW2J = delta3*(hidden.');
partialb2J = sum(delta3,2);

cost = cost / (2*m) + lambda/2*(sum(sum(W1.^2,2),1)+sum(sum(W2.^2,2),1)) + beta*KL(sparsityParam, rho_hat);
% disp('OK4');
W1grad = 1/m*partialW1J + lambda*W1;
b1grad = 1/m*partialb1J;
W2grad = 1/m*partialW2J + lambda*W2;
b2grad = 1/m*partialb2J;


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

% disp('OK');
% disp(size(W1grad));
% disp(size(W2grad));
% disp(size(b1grad));
% disp(size(b2grad));

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function kl = KL(rho, rho_hat)
    rho_sum = rho*log(rho./rho_hat) + (1-rho)*log((1-rho)./(1-rho_hat));
    kl = sum(rho_sum, 1);
end