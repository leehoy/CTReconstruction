path='D:\MachineLearning\PediMDNoNoiseTotal\LogData\';
nx=750;
ny=200;
nz=680;
list=dir(strcat(path,'*.dat'));
sino=zeros(nx,nz);
for i=1:size(list,1)
    f=fopen(strcat(path,list(i).name));
    img=fread(f,[nx ny],'float32');
    fclose(f);
    sino(:,i)=img(:,100);
end
% imshow(sino,[]);
F1=sino;