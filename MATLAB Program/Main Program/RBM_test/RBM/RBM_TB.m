function [net]=RBM_TB(x,Options)
%%
% Restricted Boltzman Machine (RBM):

% The function RBM_TB allows to train an RBM net (ristricted Boltzman machine)
% using the constrastive divergence method.
% Inputs:
% x       : training set
% Options :
% Options.eps       : learning rate
% Options.Nneurons  : number of neurons in the hidden layer
% Options.max_itera : maximum number of learning iterations
% Options.N_gs      : number of gibbs samplling steps
% Options.Sz_mb     : size of  a mnin-batch of data
% Outputs:
% net: contains important characteristics of traind net
    
    %%%%    Authors:        TAREK BERGHOUT
    %%%%    UNIVERSITY:     BATNA 2,BATNA, ALGERIA
    %%%%    EMAIL:          berghouttarek@gmail.com
    %%%%    Created: 14/04/2019
    %%%%    Updated: 29/08/2019

%% Load Options
eps=Options.eps;
Nneurons =Options.Nneurons;
N_gs=Options.N_gs;
max_itera=Options.max_itera;
%% initialization
data=scaledata(x,0,1);                                  % normalization  
I2=data;                                                % save a copy from the training data
W = (2*rand(size(data,2),Nneurons));                    % generate randomly input wiegths  W 
v_bias = zeros(1,size(data,2));                         % initial bias in visible layer (input layer)
h_bias = zeros(1,Nneurons);                             % initial bias in hidden layer
%% training processe
for i = 1:max_itera
    i
errvec=[];                                      % initialize error history for every iteration
ordering = randperm(size(data,1),Options.Sz_mb);% randomly shose a batch of data
mini_batch = data(ordering, :);                 % load our mini-Batch
    
    for j = 1:N_gs % start gibbs sampling using energy function
        
        hidden_p =  sigmoid(mini_batch * W + repmat(h_bias,size(mini_batch,1),1));         % Find hidden units by sampling the visible layer.                                           
        visible_p =sigmoid( hidden_p* W' +repmat(v_bias,size(mini_batch,1),1));   % Find visible units by sampling from the hidden ones.                                           
        bP = sigmoid(visible_p * W + repmat(h_bias,size(mini_batch,1),1));        % Last step : Find hidden units from the last visible_p.                                                  
        pD = mini_batch'*hidden_p;                             % Positive Divergence
        nD = visible_p'*bP;                           % Negative Divergence
        W = W + eps*(pD - nD);                        % update weights using contrastive divergence
        v_bias = v_bias + eps*(sum(mini_batch-visible_p));     % Update biases of the visibal layer
        h_bias = h_bias + eps*(sum(hidden_p-bP));     % Update biases of the hidden  layer
        errvec(j) =  sqrt(mse((mini_batch-visible_p)));        % Estimate error (RMSE)
    
    end
errvecT(i) = mean(errvec);%training error history
end
%% training accuracy 
Trv_bias = repmat(v_bias,size(I2,1),1);% build a biases matrix
Trh_bias = repmat(h_bias,size(I2,1),1);% build a biases matrix
Tr_h=sigmoid(I2* W + Trh_bias);        % calculated the visible layer
Tr_v=sigmoid(Tr_h* W' + Trv_bias);     % calculated the hidden layer
Tr_acc =sqrt(mse(I2-Tr_v));            % Estimate error (RMSE)

%% save trained net
net.input=I2;       % save the original normalized training data
net.regen=Tr_v;     % save the regenerated  input (reconstructed)
net.W=W;            % save updated weights weights
net.xbias=v_bias;   % save updated baises of the visible layer
net.hbias=h_bias;   % save updated baises of the hidden layer
net.Tr_acc=Tr_acc;  % save training accracy
net.hist=smooth(errvecT,13);   % save the smooth version of history of training error
end