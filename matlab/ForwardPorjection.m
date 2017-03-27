function [ output ] = ForwardProjection( image,parameters)
%Forward projection function for iterative reconstruction
%   image : reconstructed images to be forward projected
%   parameters : parameters for the forward projection (dictionary)
%   type : method to forward project images (string)
%   type can be 'ray-driven', 'pixel-driven', 'distance-driven', and
%   'separable-footprint'
type=parameters.type;
DetectorOffset=parameters.DetectorCenterOffset;
Source=parameters.InitialSource;
SAD=parameters.SourceToAxis;
SDD=parameters.SourceToDetector;
numRows=parameters.NumberOfRows;
numChan=parameters.NumberOfChannels;
pixWidth=prameters.DetectorPixelSize[0];
pixHeight=parameters.DetectorPixelSize[1];
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
  Source=[parameter.SourceToAxis*sin(-ViewAngle),parameter.SourceToAxis*cos(-ViewAngle);
  DetectorCenter=[parameters.SourceToDetector*sin(ViewAngle),parameter.SourceToDetector*cos(ViewAngle)]; % detector center is center of cntral bin
  VerticalCoordinate;
  HorizontalCooridnate;
  [CellCenterH,CellCenterV]=meshgrid(VerticalCoordinate,HorizontalCoordinate);
  CellBoundaryH1=CellCenterX;CellBoundaryH2=CellCenterX;
  CellBoundaryV1=CellCenterY;CellBoundaryV2=CellCenterY;
  CellBoundary3
  [nx,ny,nz]=parameter.ObjectSize;
  if(abs(Source(1)-DetectorCenter(1))>abs(Source(2)-DetectorCenter(2)))
      nn=nx;
      nDir=1;
  else
      nn=ny;
      nDir=2;
  end
  for i=1:nn
      if nDir==1
          k1=(Source(1)-CellBoundaryH1)/(Source(2)-DetectorCenter(2));
          k2=(Source(1)-CellBoundaryH2)/(Source(2)-DetectorCenter(2));
          Y1=floor(k1*(nx+i-0.5)+Source(2));
          Y2=floor(k2*(nx+i-0.5)+Source(2));
          k1=(Source(1)-CellBoundaryV1)/(Source(2)-DetectorCenter(2));
          k2=(Source(1)-CellBoundaryV2)/(Source(2)-DetectorCenter(2));
          Y1=floor(k1*(nx+i-0.5)+Source(2));
          Y2=floor(k2*(nx+i-0.5)+Source(2));
      else
          k1=(Source(1)-CellBoundaryH1)/(Source(2)-CellBoudaryV1);
          k2=(Source(1)-CellBoundaryH2)/(Source(2)-CellBoundaryV2);
          Y1=floor(k1*(nx+i-0.5)+Source(2));
          Y2=floor(k2*(nx+i-0.5)+Source(2));
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
