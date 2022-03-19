function [cc,alfa,it,Ire] = mean_variance_min_svm_se(K,Ire, y, c,c1,eps,pmax)
%% Solve the mean-second-order minimization with hinge loss model

%% The problem is solved by Newton-type method.
%%2020-11-12 by Cui FU. Revisted sszhou


p=100;
it=0;
[r,n]=size(K);
alfa =zeros(r,1);%start points for theta
Kalfa =zeros(r,1);
xi = ones(n,1);%r=1-y.*K*alfa
mar=max(0,xi);
tmp = exp(-p*abs(xi)); eta=mar+log(1+tmp)/p;
fun0 = sum(eta)*c + c1*mar'*mar; 

u=min(1,exp(p*xi))./(1+tmp);
grad=K*((c*u+2*c1*mar).*y/(-1)); 
itmax=300;
data=1e-10;

while (norm(grad)>eps||p<pmax)&&it<itmax
    it=it+1; 
    theta=p*tmp./((1+tmp).^2);  tao = c*theta; 
    I=logical(xi<=0);
    kk=K;
    kk(:,I)=0; 

    II = logical(tao>data);
    Hess=bsxfun(@times, K(:,II), tao(II)')*K(:,II)'+K(:,Ire)+2*c1*kk*kk'+eye(r)*(1e-4);

    d = Hess\-grad;     Kd = (d'*K)'; yKd =  Kd.*y;  
    r1 = xi - yKd;  mar=max(0,r1);  tmp = exp(-p*abs(r1)); eta=mar+log(1+tmp)/p; s=1;
    aKa = alfa'*Kalfa; aKd =  alfa'*Kd(Ire); dKd = d'*Kd(Ire); 
  
   fun1 =c*sum(eta)+ c1*mar'*mar+(aKa + 2*aKd+dKd)/2; 

    pp=0;
    while  fun1>fun0+0.05*s*grad'*d&&pp<15
        pp=pp+1;
        s=s/2;  
        yKd=yKd/2; aKd = aKd/2; dKd=dKd/4;
        r1 = xi - yKd; mar=max(0,r1); tmp = exp(-p*abs(r1)); eta=mar+1/p*log(1+tmp);
         fun1 =c*sum(eta) + c1*mar'*mar+(aKa + 2*aKd+dKd)/2; 
    end
    alfa=alfa + s*d; 
    fun0 = fun1; 
    xi=r1;     Kalfa = Kalfa +s* Kd(Ire);
    
     if  norm(grad)<max(r/p,1)&&p<pmax
       p=min(10*p,pmax);
       tmp = exp(-p*abs(r1)); eta=mar+log(1+tmp)/p;
       fun0 = c*sum(eta) + c1*mar'*mar+( alfa'*K(:,Ire)* alfa)/2 ;
     end
     
      u=min(1,exp(p*xi))./(1+tmp);
      v=(c*u+2*c1*mar).*y/(-1);  
      grad=Kalfa + K* v; %%可用tmp>10^-10提升计算效率
  %    disp(norm(grad));

end
 fun11 = sum(eta)*c +c1*mar'*mar+( alfa'*K(:,Ire)* alfa)/2; 
 funqw =mean(eta);
 v=var(eta);
 funze = ( alfa'*K(:,Ire)* alfa)/2; 
 cc=[ fun11,funqw,v,funze];
end

