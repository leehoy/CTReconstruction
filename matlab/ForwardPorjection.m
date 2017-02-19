function [ output_args ] = ForwardProjection( image,parameters, type)
%Forward projection function for iterative reconstruction
%   image : reconstructed images to be forward projected
%   parameters : parameters for the forward projection (dictionary)
%   type : method to forward project images (string)
%   type can be 'ray-driven', 'pixel-driven', 'distance-driven', and
%   'separable-footprint'
switch type
    case 'ray-driven'
    case 'pixel-driven'
    case 'distance-driven'
    case 'separable-footprint'
end



end

