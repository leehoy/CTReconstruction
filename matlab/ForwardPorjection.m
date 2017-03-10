function [ output ] = ForwardProjection( image,parameters)
%Forward projection function for iterative reconstruction
%   image : reconstructed images to be forward projected
%   parameters : parameters for the forward projection (dictionary)
%   type : method to forward project images (string)
%   type can be 'ray-driven', 'pixel-driven', 'distance-driven', and
%   'separable-footprint'
type=parameters.type;
DetectorCenter=parameters.CenterOfDetector;
Source=parameters.CenterOfSource;

switch type
    case 'ray-driven'
    case 'pixel-driven'
    case 'distance-driven'
        
    case 'separable-footprint'
end



end

