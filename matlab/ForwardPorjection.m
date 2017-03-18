function [ output ] = ForwardProjection( image,parameters)
%Forward projection function for iterative reconstruction
%   image : reconstructed images to be forward projected
%   parameters : parameters for the forward projection (dictionary)
%   type : method to forward project images (string)
%   type can be 'ray-driven', 'pixel-driven', 'distance-driven', and
%   'separable-footprint'
type=parameters.type;
DetectorOffset=parameters.CenterOfDetector;
Source=parameters.CenterOfSource;
SAD=parameters.SourceToAxis;
SDD=parameters.SourceToDetector;
DetectorCenter=
numRows=parameters.NumberOfRows;
numChan=parameters.NumberOfChannels;
pixWidth=prameters.PixelWidth;
pixHeight=parameters.PixelHeight;
NumberOfViews=parameters.NumberOfViews;
proj=zeros(numChan,numRow,NumberOfViews);
Angle=linspace(0,AngleCoverage,NumberOfViews+1);% doesn't include last point of angle coverage
Angle=Angle(1:end-1);
switch type
    case 'ray-driven'
        for i=0:NumberOfViews
            proj(:,:,i)=ray(image,paraeters,Angle(i));
        end
    case 'pixel-driven'
        for i=0:NumberOfViews
            proj(:,:,i)=pixel(image,paraeters,Angle(i));
        end
    case 'distance-driven'
        for i=0:NumberOfViews
            proj(:,:,i)=distance(image,paraeters,Angle(i));
        end
    case 'separable-footprint'
        for i=0:NumberOfViews
            proj(:,:,i)=separablefootprint(image,paraeters,Angle(i));
        end
end



end
function distance(image,parameter,ViewAngle)
VerticalCoordinate;
HorizontalCooridnate;
[CellCenterX,CellCenterY]=meshgrid(VerticalCoordinate,HorizontalCoordinate);
CellBoundary1X=CellCenterX;CellBoudnary1Y=CellCenterY;
CellBoundary2X=CellCenterX;
CellBoundary2Y=CellCenterY;
CellBoundary3
[nx,ny,nz]=parameter.ObjectSize;
Source=parameter.CenterOfSource;
DetectorCenter;
if(abs(Source(1)-DetectorCenter(1))>abs(Source(2)-DetectorCenter(2)))
    nn=nx;
    nDir=1;
else
    nn=ny;
    nDir=2;
end
for i=1:nn
    if nDir==1
        k1=(Source(2)-)/(Source(1)-);
        k2=(Source(2)-)/(Source(1)-);
        Y1=floor(k1*(nx+i-0.5)+Source(2));
        Y2=floor(k2*(nx+i-0.5)+Source(2));
    else
    end
    if (Y1==Y2 && Z1==Z2)
    elseif(Y1==Y2 && Z1~=Z2)
    elseif(Y1~=Y2 && Z1==Z2)
    else
    end
end
end
function ray(image,parameter,ViewAngle)
end
function pixel(image,parameter,ViewAngle)
end
function separablefootprint(image,parameter,ViewAngle)
end