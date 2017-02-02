function [ g] = FanFilter( x,a)
%UNTITLED5 이 함수의 요약 설명 위치
%   자세한 설명 위치
g=zeros(length(x),1);
% a=(a/360)*2*pi;
h=Filter(x,a);
c=1;
for i=1:length(x)
    if(x(i)==0)
%         g(c)=1/(8*a^2);
        g(i)=0.5*h(i);
%         c=c+1;
    elseif(mod(abs(x(i)),2)==0)
        g(i)=0;
%         c=c+1;
    else
%         a=a*2*pi/360;
%         g(c)=-0.5*(a/(pi*a*sin(i*a)))^2;
        g(i)=0.5*h(i)*((x(i)*a)/(sin(x(i)*a)))^2;
%         g(c)=h(c)*0.5*((i*a)/(sin(i*a)))^2;
%         c=c+1;
    end
end
end
 
