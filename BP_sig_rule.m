%加载数据集
clc;clear;close all;
data=readmatrix('data\abalone.csv');

[m,n]=size(data);
train_num=round(0.8*m);
test_num=m-train_num;

x_train=data(1:train_num,1:n-1);
y_train=data(1:train_num,n);
x_test=data(train_num+1:m,1:n-1);
y_test=data(train_num+1:m,n);

%初始化参数
Neurous_num=6;  %隐藏神经元个数
input_num=n-1;
output_num=1;
[x_train_std,x_train_mu,x_train_sigma]=zscore(x_train);
[y_train_std,y_train_mu,y_train_sigma]=zscore(y_train);

x_test_std=(x_test-repmat(x_train_mu,test_num,1))./repmat(x_train_sigma,test_num,1);    %用训练集参数标准化测试集
x_train_std=x_train_std';
x_test_std=x_test_std';
y_train_std=y_train_std';


%网络参数
vij=rand(Neurous_num,input_num);
theta_u=rand(Neurous_num,1);
wj=rand(output_num,Neurous_num);
theta_y=rand(output_num,1);

learn_rate=0.0001;  %学习率
Epochs_max=50000;    %最大迭代数
error_rate=0.1;
Obj_save=zeros(1,Epochs_max);

%训练网络
%%误差分析
epoch_num=0;
while epoch_num<Epochs_max
    epoch_num=epoch_num+1;
    
    y_pre_std_u=vij*x_train_std+repmat(theta_u,1,train_num);
    y_pre_std_u1=logsig(y_pre_std_u);

    y_pre_std_y=wj*y_pre_std_u1+repmat(theta_y,1,train_num);
    y_pre_std_y1=y_pre_std_y;

    obj=y_pre_std_y1-y_train_std;
    Ems=sumsqr(obj);
    Obj_save(epoch_num)=Ems;

    if Ems<error_rate
        break;
    end
%%梯度下降
%%%输出采用rule函数，隐藏层采用sigomd激活函数    
    c_wj=2*(y_pre_std_y1-y_train_std)*y_pre_std_u1';
    c_theta_y=2*(y_pre_std_y1-y_train_std)*ones(train_num,1);
    c_vij=wj'*2*(y_pre_std_y1-y_train_std).*y_pre_std_u1.*(1-y_pre_std_u1)*x_train_std';
    c_theta_u=wj'*2*(y_pre_std_y1-y_train_std).*y_pre_std_u1.*(1-y_pre_std_u1)*ones(train_num,1);
    
    wj=wj-learn_rate*c_wj;
    theta_y=theta_y-learn_rate*c_theta_y;
    vij=vij-learn_rate*c_vij;
    theta_u=theta_u-learn_rate*c_theta_u;

end

%使用模型
test_put=logsig(vij*x_test_std+repmat(theta_u,1,test_num));
test_out=wj*test_put+repmat(theta_y,1,test_num);

%反归一化
test_pre_out=test_out*y_train_sigma+y_train_mu;
errors_nn=sum(abs((test_pre_out'-y_test)./y_test))/test_num;
%errors_nn=sum(abs((test_pre_out'-y_test)./y_test))/length(y_test);
%画图
figure(1);
plot(Obj_save,'b-','LineWidth',1.5);
title('损失函数');
xlabel('epoch');
ylabel('errors');
figure(2);
color=[111,168,86;128,199,252;112,138,248;184,84,246;]/255;
plot(y_test,'Color',color(2,:),'LineWidth',1);
hold on;
plot(test_pre_out,'*','Color',color(1,:));
hold on;
titlestr=['公式推导BP神经网络','误差为：',num2str(errors_nn)];
title(titlestr);
