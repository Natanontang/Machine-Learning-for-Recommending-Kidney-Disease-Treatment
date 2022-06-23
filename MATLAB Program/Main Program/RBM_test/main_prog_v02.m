%%I dedicate this work To my son "BERGHOUT Loukmane"close all;
clear all;
clc
addpath('RBM');
%% load training data
% Note: for this toy example, each data sample = 16 samples
% sample index 1-4  = disease (1000,0100,0010,0001)
% sample index 5-16 = drug (5-7 for 1000,8-10 for 0100,11-13 for 0010 and
%                           14-16 for 0001)
% prepare training data
% pathname        = uigetdir;
% allfiles        = dir(fullfile(pathname,'*.txt'));
% xtr=[];       % initialize training inputs
% for i=1:size(allfiles,1)    
%     x=load([pathname '\\' allfiles(i).name]);
%     x=double(x);
%     xtr=[xtr; x];% training set building
% end
%% Import data:
training_set = readtable('Training_set_80.csv');
testing_set = readtable('Testing_set_20.csv');
%drop col:
training_set(:,1) = [];
testing_set(:,1) = [];
%convert table to array
xtr = table2array(training_set);
xtest = table2array(testing_set);
%% Training Options
Options.max_itera=717*3;        % maximum number of learning itterations for 32:iteration 
Options.N_gs=60;                  % number of gibbs samplling steps
Options.Nneurons=15; %gamma(1);    % number of neurons in the hidden layer
Options.eps=0.005;                 % learning rate
Options.Sz_mb=16;                 % size if mini-batch of data
%% Training process 
net=RBM_TB(xtr,Options);% training
net.Tr_acc
%% Prediction
%Note: for this example, sample index 1-4 of testing data is a subset of
%sample index 1-4 training data ==> this is just toy example
%----------------------------------------------------------------------%
%xp=load('Testing_data01.txt');
% xp=zeros(1,16);
% xp(5:16)=-1
% xp(4)=1;%set xp(t)=1 for disease, t=1(1000),2(0100),3(0010),4(0001) 
%----------------------------------------------------------------------%
%เอา Training set มา Test:
% select_row = 1;
% actual = xtr(select_row,:);
% xp = xtr(select_row,:);
% xp(13:114)= 0;

%เอา Testing set มา Test:
select_row = 600; %ex:600 %2882
actual = xtest(select_row,:);
xp = xtest(select_row,:);
xp(13:114)= 0; %ex:600 ถ้าให้ Drug = 0 ผลจะมีตัว overlap ยาจริงที่จ่ายเยอะกว่า, ถ้าให้ Drug = -1 ผลจะมีตัว overlap ยาจริงน้อยกว่า
%----------------------------------------------------------------------%
[y]=RBM_RECONSTRUCT(net,xp); %check index 5-16 of y for drug suggestion
% Illustration 

figure(1)
subplot(2,2,1)
stem(y)
grid on
title('regenerated input')

subplot(2,2,2)
stem(xp);
grid on
title('input')

subplot(2,2,3:4)
stem(actual);
grid on
title('actual')

figure(3)
subplot(2,2,1:2)
plot(1:length(net.hist),net.hist,'LineWidth',2)
xlabel('number of iterations')
ylabel('RMSE of training')
grid on


