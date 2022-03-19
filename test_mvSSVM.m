function [accuracy]=test_mvSSVM(x,xnorm,alfa,xe,te_label,h,Ire)
%x≤‚ ‘ ˝æ›£¨y≤‚ ‘±Í«©
xenorm=sum(xe.*xe,2);
Kte = bsxfun(@plus,xnorm(Ire),xenorm') - 2* x(Ire,:)*xe';        Kte = exp(Kte*(-h));%%rbf kenrnl
accuracy=1-sum(te_label.*(alfa'*Kte)'<0)/length(te_label);
end