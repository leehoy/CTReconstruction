function [ filter ] = FilterArc( N,tau,cutoff,smooth )
%Create spatial filter for reconstruction
%   returning spatial filter for filtered back projection
%   takes number of pixel spacing and number channel as input

a=tau;
g=zeros(N,1);
x=-floor((N-1)/2):floor((N-1)/2);
% g(x==0)=1/(8*a^2);
g(x==0)=1/(8*a^2);
odds= find(mod(x,2)==1);
% g(odds)=-1./(2*pi^2*a^2*x(odds).^2);
g(odds)=-1./(2*pi^2*a^2*sin(x(odds)*a).^2);
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
filter(abs(w)>pi*cutoff/(2*a))=0;
% filter=filter*0.5;

end

 
