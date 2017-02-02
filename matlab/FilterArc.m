function [ h ] = FilterArc( N,tau,cutoff,smooth )
%Create spatial filter for reconstruction
%   returning spatial filter for filtered back projection
%   takes number of pixel spacing and number channel as input

assert(mod(N,2)==1,'Number of pixel should be odd number');
x=(1:N)-(N-1)/2;
h1=FanFilter(x,tau);
h2=FanFilter(x-0.5/cutoff,tau);
h3=FanFilter(x+0.5/cutoff,tau);
h=smooth*h1+(1-smooth)/2*(h2+h3);
end

 