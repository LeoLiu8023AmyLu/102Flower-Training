clc;clear;
tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%以下为训练部分
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Files = dir(fullfile('D:\Flower\train\','*.jpg'));
LengthFiles = length(Files);

height=50;width=50;%特征向量大小，可以调整
w1 = zeros(height*width,LengthFiles/3);%第一类训练矩阵
w2 =  zeros(height*width,LengthFiles/3);%第二类训练矩阵
w3 =  zeros(height*width,LengthFiles/3);%第三类训练矩阵

%第一类图像
for i = 1:LengthFiles/3
    img = imread(strcat('D:\Flower\train\',Files(i).name));%读取图像
    img1 = rgb2gray(img);%转换成灰度图
    if i>=1 && i<=5
    figure;imshow(img1)
    end
    [m,n] = size(img1);
    m1 = round(m/2);n1=round(n/2);
    img11 = img1(m1-199:m1+200,n1-199:n1+200);%截取中间图像

    img2 = imresize(img11,[height,width]);%调整大小
    w1(:,i) = double(img2(:)); %得到训练向量
end
%第二类图像
for i = LengthFiles/3+1:2*LengthFiles/3
    img =imread(strcat('D:\Flower\train\',Files(i).name));
    img1 = rgb2gray(img);
    if i>=1 && i<=5
    figure;imshow(img1)
    end
    [m,n] = size(img1);
    m1 = round(m/2);n1=round(n/2);
    img11 = img1(m1-199:m1+200,n1-199:n1+200);
    img2 = imresize(img11,[height,width]);
    w2(:,i-LengthFiles/3) = double(img2(:)); 
end
%第三类图像
for i =2*LengthFiles/3+1:LengthFiles
    img = imread(strcat('D:\Flower\train\',Files(i).name));
    img1 = rgb2gray(img);
    if i>=1 && i<=5
    figure;imshow(img1)
    end
    [m,n] = size(img1);
    m1 = round(m/2);n1=round(n/2);
    img11 = img1(m1-199:m1+200,n1-199:n1+200);
    img2 = imresize(img11,[height,width]);
    w3(:,i-2*LengthFiles/3) = double(img2(:)); 
end

%ｓｖｍ训练
w12 = [w1 w2];
w13 = [w1 w3];
w23 = [w2 w3];
Y =[ones(1,LengthFiles/3) -1*ones(1,LengthFiles/3)];
svmStruct12 = svmtrain(w12',Y,'Kernel_Function','quadratic');%区分第一类和第二类
svmStruct13 = svmtrain(w13',Y,'Kernel_Function','quadratic');%区分第一类和第三类
svmStruct23 = svmtrain(w23',Y,'Kernel_Function','quadratic');%区分第二类和第二类

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%以下为测试部分
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Files = dir(fullfile('D:\Flower\test\','*.jpg'));
LengthFiles = length(Files);
correct=0;
for i=1:30
    img = imread(strcat('D:\Flower\test\',Files(i).name));
    img1 = rgb2gray(img);
    [m,n] = size(img1);
    m1 = round(m/2);n1=round(n/2);
    img11 = img1(m1-199:m1+200,n1-199:n1+200);
    img2 = imresize(img11,[height,width]);
    testy = double(img2(:)); %训练向量
    
    result12 = svmclassify(svmStruct12,testy');%大于零为第一类，小于零为第二类
    result23 = svmclassify(svmStruct23,testy');%大于零为第二类，小于零为第三类
    result13 = svmclassify(svmStruct13,testy');%大于零为第一类，小于零为第三类

    
    if(result12>0&result13>0)
        correct=correct+1;  
   end
end
for i=31:60
    img = imread(strcat('D:\Flower\test\',Files(i).name));
    img1 = rgb2gray(img);
    img2 = imresize(img1,[height,width]);
    testy = double(img2(:)); 
    
    result12 = svmclassify(svmStruct12,testy');
    result23 = svmclassify(svmStruct23,testy');
    result13 = svmclassify(svmStruct13,testy');

    
     if(result12<0&result23>0)
           correct=correct+1;  
     
    end
end
for i=61:LengthFiles
    img = imread(strcat('D:\Flower\test\',Files(i).name));
    img1 = rgb2gray(img);
    img2 = imresize(img1,[height,width]);
    testy = double(img2(:)); 
    
    result12 = svmclassify(svmStruct12,testy');
    result23 = svmclassify(svmStruct23,testy');
    result13 = svmclassify(svmStruct13,testy');

    
    if(result13<0&result23<0)
        correct=correct+1;  
      
    end
end

%计算识别率
result = correct/LengthFiles*2.5;
disp(['SVM方法识别率为 ',num2str(result)])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

toc





