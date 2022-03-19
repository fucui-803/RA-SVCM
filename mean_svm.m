function [cc, alfa,it,Ire] = mean_svm(K,Ire, y, c,eps,pmax)
%% Solve the mean-minimization with hinge loss model
%% The model is coming from JMLR 2018(18),pp1-23, Accelerating Stochastic Composition Optimization
%% Here we try hinge loss for regression  or classification
%% The input x is the training samples in row, y is its target.
%% If y is a real vector, it is regression problem; else if y is a logical vector, it is for classification.
%% The problem is solved by Newton-type method.
% %%2020-11-12 by Cui FU. Revisted sszhou
p=1;
it=0;
[r,n]=size(K);
alfa =zeros(r,1);%start points for theta
Kalfa =zeros(r,1);
xi = ones(n,1);%r=1-y.*K*alfa
tmp = exp(-p*abs(xi)); eta=max(0,xi)+log(1+tmp)/p; 
fun0 = sum(eta)*c ;
u=min(1,exp(p*xi))./(1+tmp);
grad=-K*(c*u.*y);
itmax=300;
data=1e-10;

while  (norm(grad)>eps||p<pmax)&&it<itmax
    it=it+1; 
    theta=p*tmp./((1+tmp).^2);      tao = theta*c;
    II = logical(tao>data);
    Hess=bsxfun(@times, K(:,II), tao(II)')*K(:,II)'+K(:,Ire)+eye(r)*(1e-4);
    
    d = Hess\-grad;    Kd = (d'*K)';    yKd =  Kd.*y;  
    r1 = xi - yKd;    tmp = exp(-p*abs(r1)); eta=max(0,r1)+log(1+tmp)/p; s=1;
    aKa = alfa'*Kalfa; aKd =  2*alfa'*Kd(Ire); dKd = d'*Kd(Ire); 
    fun1 = sum(eta)*c +(aKa +aKd+dKd)/2; 
  
    pp=0;
    while  fun1>fun0+0.05*s*grad'*d&&pp<15
        pp=pp+1;
        s=s/2;
        yKd=yKd/2; aKd = aKd/2; dKd=dKd/4;
        r1 = xi - yKd;  tmp = exp(-p*abs(r1)); eta=max(0,r1)+log(1+tmp)/p;
        fun1 = sum(eta)*c +(aKa+aKd+dKd)/2;
    end  
    alfa=alfa + s*d; 
    fun0 = fun1;
    xi=r1;     Kalfa = Kalfa +s* Kd(Ire);
    %%可用tmp>10^-10提升计算效率
   if  norm(grad)<max(r/p,1)&&p<pmax
       p=min(10*p,pmax);
       tmp = exp(-p*abs(xi));  eta=max(0,xi)+log(1+tmp)/p;
       fun0 = sum(eta)*c +(alfa'*K(:,Ire)*alfa)/2;      
   end 
     u=min(1,exp(p*xi))./(1+tmp); 
     grad=Kalfa - K*(u.*y)*c ; 
%       disp(norm(grad));
   
end
fun11 = sum(eta)*c +( alfa'*K(:,Ire)* alfa)/2 ;
funqw = sum(eta)/n;
v=var(eta);
funze = ( alfa'*K(:,Ire)* alfa)/2; 
cc=[fun11,funqw,v,funze];
end






