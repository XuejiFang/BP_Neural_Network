clc;clear;close all;
data=readmatrix('data\abalone.csv');

[m,n]=size(data);
train_num=round(0.8*m);
test_num=m-train_num;

x_train=data(1:train_num,1:n-1);
y_train=data(1:train_num,n);
x_test=data(train_num+1:m,1:n-1);
y_test=data(train_num+1:m,n);

x_train=x_train';
y_train=y_train';
x_test=x_test';

[x_train_regular,x_train_maxmin]=mapminmax(x_train);
[y_train_regular,y_train_maxmin]=mapminmax(y_train);

%创建网络
%%调用形式
%net=newff(x_train_regular,y_train_regular,[6,5],{'logsig','tansig','purelin'});
%net=newff(x_train_regular,y_train_regular,[6,3,3],{'logsig','tansig','logsig','purelin'});
%net=newff(x_train_regular,y_train_regular,6,{'logsig','logsig'});
%net=newff(x_train_regular,y_train_regular,6,{'logsig','purelin'});
net=newff(x_train_regular,y_train_regular,6,{'logsig','tansig'});
%%设置迭代次数
%net.trainParam.epochs=50000;
%%设置收敛误差
%net.trainParam.goal=0.000001;

%训练网络
[net,~]=train(net,x_train_regular,y_train_regular);
%将输出数据归一化
x_test_regular=mapminmax('apply',x_test,x_train_maxmin);
%放入到网络输出数据
y_test_regular=sim(net,x_test_regular);
% 将得到的数据反归一化得到预测数据
BP_pre=mapminmax('reverse',y_test_regular,y_train_maxmin);

BP_pre=BP_pre';
errors_nn=sum(abs(BP_pre-y_test)./(y_test))/length(y_test);
figure(1);
color=[111,168,86;128,199,252;112,138,248;184,84,246;]/255;
plot(y_test,'Color',color(2,:),'LineWidth',1);
hold on;
plot(BP_pre,'*','Color',color(1,:));
hold on
titlestr=['MATLAB自带BP神经网络','误差为：',num2str(errors_nn)];
title(titlestr);