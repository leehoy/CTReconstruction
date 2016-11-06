function [ h ] = FilterLine( N,tau,cutoff,smooth )
%Create spatial filter for reconstruction
%   returning spatial filter for filtered back projection
%   takes number of pixel spacing and number channel as input

assert(mod(N,2)==1,'Number of pixel should be odd number');
x=(1:N)-(N-1)/2;
h1=Filter(x,tau,cutoff);
h2=Filter(x-0.5/cutoff,tau,cutoff);
h3=Filter(x+0.5/cutoff,tau,cutoff);
h=smooth*h1+(1-smooth)/2*(h2+h3);
end

