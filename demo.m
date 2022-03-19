clc;clear;
load('adult.mat');
x=full(tr_instance); xe=full(te_instance);
espl=1e-3;
h=2^(-6);
p=100;
n=size(x,1);   
%%sRA-SVM
lamda1= 1e-1;
lamda2= 1e0;
%%RA-SVM
Lamda1= 1e0;
Lamda2= 1e1;
%
%%SVM
Lamda= 1e1;
r = min(max(100,n*0.1),800); Ire=randperm(n); Ire=Ire(1:r);%r--reduced size
%º∆À„∫À∫Ø ˝
xnorm = sum(x.^2,2);
K = bsxfun(@plus,xnorm(Ire),xnorm')-2* x(Ire,:)*x';        K= exp(K*(-h));%%rbf kenrnl
%sRA-SVM
 [cc1,alfa1,it1,Ire1] = mean_variance_min_svm_se(K,Ire,  tr_label, lamda1,lamda2,espl,p);

 [cc2,alfa2,it2,Ire2] = mean_variance_min_svm(K,Ire,  tr_label, Lamda1,Lamda2,espl,p);  
 
 [cc3,alfa3,it3,Ire3] = mean_svm(K,Ire, tr_label,Lamda,espl,p);
%º∆À„—µ¡∑¥ÌŒÛ¬ 
Tr_err1=sum(tr_label.*(alfa1'*K)'<0)/length(tr_label);
Tr_err2=sum(tr_label.*(alfa2'*K)'<0)/length(tr_label);
Tr_err3=sum(tr_label.*(alfa3'*K)'<0)/length(tr_label);
%º∆À„≤‚ ‘¥ÌŒÛ¬ 
xenorm=sum(xe.*xe,2);
Kte = bsxfun(@plus,xnorm(Ire),xenorm') - 2* x(Ire,:)*xe';        Kte = exp(Kte*(-h));%%rbf kenrnl
Te_err1=sum(te_label.*(alfa1'*Kte)'<0)/length(te_label);
Te_err2=sum(te_label.*(alfa2'*Kte)'<0)/length(te_label);
Te_err3=sum(te_label.*(alfa3'*Kte)'<0)/length(te_label);

disp(['SVM:',' ','bestacc=',num2str(mean(Te_err3))]);
disp(['RA-SVM:',' ','bestacc=',num2str(mean(Te_err2))]);
disp(['sRA-SVM:',' ','bestacc=',num2str(mean(Te_err1))]);
