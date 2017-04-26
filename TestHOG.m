clear;clc;

img=imread('C:\Users\Anshare_LY\Pictures\4M6A0713-蓝色之星60X120.JPG');

% 1、%灰度化
img=rgb2gray(img);
img=double(img);

step=8;      %step*step个像素作为一个cell
[m1 n1]=size(img);
%改变图像尺寸为step的最近整数倍
img=imresize(img,[floor(m1/step)*step,floor(n1/step)*step],'nearest');
[m n]=size(img);

% 2、%伽马校正
img=sqrt(img);

% 3、求梯度和方向
fy=[-1 0 1];        %定义竖直模板
fx=fy';             %定义水平模板
Iy=imfilter(img,fy,'replicate');    %竖直梯度
Ix=imfilter(img,fx,'replicate');    %水平梯度
Ied=sqrt(Ix.^2+Iy.^2);              %梯度值
Iphase=Iy./Ix;              %边缘斜率，有些为inf,-inf,nan，其中nan需要再处理一下
the=atan(Iphase)*180/3.14159; %求梯度角度

for i=1:m
    for j=1:n
        if(Ix(i,j)>=0&Iy(i,j)>=0) %第一象限
            the(i,j)=the(i,j);
        elseif(Ix(i,j)<=0&Iy(i,j)>=0) %第二象限
            the(i,j)=the(i,j)+180;
        elseif(Ix(i,j)<=0&Iy(i,j)<=0) %第三象限
            the(i,j)=the(i,j)+180;
        elseif(Ix(i,j)>=0&Iy(i,j)<=0) %第四象限
            the(i,j)=the(i,j)+360;
        end

        if isnan(the(i,j))==1  %0/0会得到nan，如果像素是nan，重设为0
            the(i,j)=0;
        end

    end
end
the=the+0.000001; %防止角度为0

% 4、划分cell，求cell的直方图( 1 cell = 8*8 pixel )
clear i j;
%下面是求cell
step=8;                %step*step个像素作为一个cell
orient=9;               %方向直方图的方向个数
jiao=360/orient;        %每个方向包含的角度数
Cell=cell(1,1);              %所有的角度直方图,cell是可以动态增加的，所以先设了一个
ii=1;
jj=1;

for i=1:step:m
    ii=1;
    for j=1:step:n
        Hist1(1:orient)=0;
        for p=1:step
            for q=1:step
                %梯度方向直方图
                Hist1(ceil(the(i+p-1,j+q-1)/jiao))=Hist1(ceil(the(i+p-1,j+q-1)/jiao))+Ied(i+p-1,j+q-1);
            end
        end
        Cell{ii,jj}=Hist1;       %放入Cell中
        ii=ii+1;
    end
    jj=jj+1;
end

% 5、划分block，求block的特征值,使用重叠方式( 1 block = 2*2 cell )
clear m n i j;
[m n]=size(Cell);
feature=cell(1,(m-1)*(n-1));
for i=1:m-1
    for j=1:n-1
        block=[];
        block=[Cell{i,j}(:)' Cell{i,j+1}(:)' Cell{i+1,j}(:)' Cell{i+1,j+1}(:)'];
        block=block./sum(block); %归一化
        feature{(i-1)*(n-1)+j}=block;
    end
end

% 6、图像的HOG特征值
[m n]=size(feature);
l=2*2*orient;
featureVec=zeros(1,n*l);
for i=1:n
    featureVec((i-1)*l+1:i*l)=feature{i}(:);
end

figure
mesh(featureVec)