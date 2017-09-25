function [ filter] = FilterLine( N,a,type,cutoff)
%UNTITLED5 이 함수의 요약 설명 위치
%   자세한 설명 위치
g=zeros(N,1);
x=-floor((N-1)/2):floor((N-1)/2);
g(x==0)=1/(8*a^2);
odds= find(mod(x,2)==1);
g(odds)=-1./(2*pi^2*a^2*x(odds).^2);
g=g(1:end-1);
filter=abs(fftshift(fft(g)));
w=2*pi*x(1:end-1)./(N-1);
w=w';
switch lower(type)
    case 'ram-lak'
        %Do nothing
    case 'shepp-logan'
        zero=find(x==0);
        tmp=filter(zero); % Avoid zero-division
        filter=filter.*sin(w./(2*cutoff))./(w./(2*cutoff));
        filter(zero)=tmp;
    case 'cosine'
        filter=filter.*cos(w./(2*cutoff));
    case 'hamming'
        filter=filter.*(0.54+0.46*cos(w./cutoff));
    case 'hann'
        filter=filter.*(0.5+0.5*cos(w./cutoff));
    otherwise
        error('Wrong filter selection')
end
filter(abs(w)>pi*cutoff)=0;
end

 