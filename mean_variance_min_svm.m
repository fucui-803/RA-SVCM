function [cc,alfa,it,Ire] = mean_variance_min_svm(K,Ire, y, c,c1,eps,pmax)
%% Solve the mean-variance minimization with hinge loss model
%% The model is coming from JMLR 2018(18),pp1-23, Accelerating Stochastic Composition Optimization
%% Here we try hinge loss for regression  or classification
%% The input x is the training samples in row, y is its target.
%% If y is a real vector, it is regression problem; else if y is a logical vector, it is for classification.
%% The problem is solved by Newton-type method.
%%2020-11-12 by Cui FU. Revisted sszhou

p=1;
it=0;
[r,n]=size(K);
lambda=c*n;lambda1=c1*n;
alfa =zeros(r,1);%start points for theta
Kalfa =zeros(r,1);
xi = ones(n,1);%r=1-y.*K*alfa
tmp = exp(-p*abs(xi)); eta=max(0,xi)+log(1+tmp)/p;
fun0 = sum(eta)*(lambda/n) + lambda1/n*(eta'*eta-sum(eta)^2/n); 
u=min(1,exp(p*xi))./(1+tmp);
grad=K*((lambda-2*lambda1*sum(eta)/n+2*lambda1*eta).*u.*y)/(-n); 
itmax=300;
data=1e-10;
while (norm(grad)>eps||p<pmax)&&it<itmax
    theta=p*tmp./((1+tmp).^2);  
    tao = ((lambda-2*lambda1*sum(eta)/n+2*lambda1*eta).*theta/n+2*lambda1/n*u.*u); 
    q = K*(u.*y);
    II = logical(tao>data);
    Hess=bsxfun(@times, K(:,II), tao(II)')*K(:,II)'+K(:,Ire)- (2*lambda1/n/n*q)*q'+eye(r)*(1e-4);
    d = Hess\-grad;     Kd = (d'*K)'; yKd =  Kd.*y;  
    r1 = xi - yKd;    tmp = exp(-p*abs(r1)); eta=max(0,r1)+log(1+tmp)/p; s=1;
    aKa = alfa'*Kalfa; aKd =  alfa'*Kd(Ire); dKd = d'*Kd(Ire); 
   fun1 = sum(eta)*(lambda/n) + lambda1*(eta'*eta-sum(eta)^2/n)/n+(aKa + 2*aKd+dKd)/2; 

    pp=1;
    while  fun1>fun0+0.05*s*grad'*d&&pp<15
        s=s/2;  
        yKd=yKd/2; aKd = aKd/2; dKd=dKd/4;
        r1 = xi - yKd;  tmp = exp(-p*abs(r1)); eta=max(0,r1)+1/p*log(1+tmp);
        fun1 = sum(eta)*(lambda/n) + lambda1*(eta'*eta-sum(eta)^2/n)/n+(aKa + 2*aKd+dKd)/2;
   pp=pp+1;
    end

    alfa=alfa + s*d; 
    fun0 = fun1; 

 
    xi=r1;     Kalfa = Kalfa +s* Kd(Ire);
         
     if  norm(grad)<max(r/p,1)&&p<pmax
       p=min(10*p,pmax);
       %data=data/10;max(r/p,1)
       tmp = exp(-p*abs(r1)); eta=max(0,r1)+log(1+tmp)/p;
       fun0 = sum(eta)*(lambda/n) + lambda1*(eta'*eta-sum(eta)^2/n)/n+( alfa'*K(:,Ire)* alfa)/2 ;
     end
     
      u=min(1,exp(p*xi))./(1+tmp);
      v = (lambda-2*lambda1*sum(eta)/n+2*lambda1*eta)/n.*u.*y;
      grad=Kalfa - K* v; 
%       disp(norm(grad));
     it=it+1; 
end

 fun11 = sum(eta)*(lambda/n) + lambda1*(eta'*eta-sum(eta)^2/n)/n+( alfa'*K(:,Ire)* alfa)/2 ;
 funqw =mean(eta);
 v=var(eta);
 funze = ( alfa'*K(:,Ire)* alfa)/2; 
 cc=[ fun11,funqw,v,funze];
end





