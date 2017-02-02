function  ReconParams=ConeBeamInit( filename)
%Cone-Beam reconstruction initializer
%   The file name in input argument contains information of reconstruction

try
    key={'DataPath','SourceToAxis','SourceToDetector','DetectorPixelWidth',...
        'DetectorPixelHeight','DetectorWidth','DetectorHeight','NumberOfViews',...
        'AngleCoverage','precision','ReconX','ReconY','ReconZ'};
    value=cell(size(key));
    ReconParams=containers.Map(key,value);
    f=fopen(filename);
    while ~feof(f)
        l=fgets(f);
        tmp=strsplit(strtrim(l),':');
        if(strcmp(tmp{1},'DataPath') || strcmp(tmp{1},'precision'))
            ReconParams(tmp{1})=tmp{2};
        elseif(strcmp(tmp{1},'ReconVolume'))
            tttt=strsplit(tmp{2},'*');
            ReconParams('ReconX')=str2double(tttt{1});
            ReconParams('ReconY')=str2double(tttt{2});
            ReconParams('ReconZ')=str2double(tttt{3});
        elseif(strcmp(tmp{1},'AngleCoverage'))
            tttt=str2double(tmp{2})*pi/180;
            ReconParams('AngleCoverage')=tttt;
        else
            ReconParams(tmp{1})=str2double(tmp{2});
        end
    end
    fclose(f);
    
catch ME
end

 
end

