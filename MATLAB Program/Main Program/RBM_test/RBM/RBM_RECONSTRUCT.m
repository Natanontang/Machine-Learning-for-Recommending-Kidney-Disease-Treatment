function [y,y2]=RBM_RECONSTRUCT(net,x)
% net : The trained RBM
% x   : Unlabled data for reconstruction
% y   : reconstructed data
% y2  : encoded data
%% normalize data
x=scaledata(x,0,1); 
%% load training parameters
W=net.W; % weights 
v_bias=net.xbias;% biases of the visible layer
h_bias=net.hbias;% biases of the hidden layer
%% prediction
Tsv_bias = repmat(v_bias,size(x,1),1);% build a matrix of biases
Tsh_bias = repmat(h_bias,size(x,1),1);% build a matrix of biases
Ts_h=sigmoid(x* W + Tsh_bias);        % extruct features
Ts_v=sigmoid(Ts_h* W' + Tsv_bias);    % rebuild features
y=Ts_v;
y2=Ts_h;
end