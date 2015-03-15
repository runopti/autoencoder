function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  

% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

EPS = 10E-4;
[size_ n] = size(theta); %size_ is supposed to be 64*25*2+(64+25)*2 = 3289
disp(size_);
for i = 1:size_
    temp = eye(size_);
    e_i = temp(:,i);
    theta_iPlus = theta + EPS*e_i;
    theta_iMinus = theta - EPS*e_i;
%     disp('OKKK');
    numgrad(i) = (J(theta_iPlus)-J(theta_iMinus)) / (2*EPS);
end
% disp('OKK');
%% ---------------------------------------------------------------
