clc;clear;
clear all
close all
addpath('./RF_Class_C/'); % 添加路径
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%以下为随机森林算法预处理部分
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmpi(computer,'PCWIN') | strcmpi(computer,'PCWIN64')
   compile_windows
else
   compile_linux
end
tic
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%以下为程序控制部分     你要设置的
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Flag_Show_Picture=0;    % 控制是否显示图片                1 显示 0 不显示
Flag_Use_Random_Index=0;% 控制是否使用随机引索            1 使用 0 不使用
Flag_Test_Detail=0;     % 是否计算每个测试图片的识别细节  1 显示 0 不显示
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%以下为初始化设置部分   你要设置的 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
File_Fofer_Path='D:\Flower\';   % 文件夹目录 直接改这里就可以了 其他地方完全不用改，自动的
Original_Set='flower102\';% 所有Flower图片 目录 
Flower_Num=3;     % 选择读取花图片的种类数量 ***重点设置项目
Train_Set_Num=30;   % 选择作为训练的图片数量  ***重点设置项目
Test_Set_Num=20;    % 选择作为检测的图片数量  ***重点设置项目
height=128;width=128;% 特征向量大小，可以调整
Picture_Cut_Size=400;% 图片截取大小，可以调整 默认:400 
J_Rate=0.9;     % 判断是否为视为可识别的比率 这个设置越低 正确率越高 ***重点设置项目
%% RF 参数
RF_Tree= 0*2*Flower_Num*Train_Set_Num;       % 设置 随机森林算法中 树的数量   ***重点设置项目
mtry = 0*max(floor(height*width/3),1);          % 设置 随机森林算法中 分裂的数量   ***重点设置项目
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%以下为初始化计算部分
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 显示参数
disp('本次 随机森林 算法 参数:')
disp(['-->训练对象为 ',num2str(Flower_Num),' 种不同种类的花朵图片'])
disp(['-->每种花中选取 ',num2str(Train_Set_Num),' 张图片作为训练集'])
disp(['-->每种花中选取 ',num2str(Test_Set_Num),' 张图片作为测试集'])
disp(['-->随机森林树的数量: ',num2str(RF_Tree)])
disp(['-->随机森林分裂的数量: ',num2str(mtry)])
disp(['-->特征向量高度: ',num2str(height)])
disp(['-->特征向量宽度: ',num2str(width)])
disp(['-->图像截取宽度(正方形): ',num2str(Picture_Cut_Size)])
disp(['-->判断阈值: ',num2str(J_Rate),' (判断是否视为识别正确的比率)'])
load([File_Fofer_Path,'imagelabels.mat']);% 读取图片标签(对应每张图片的分类) 文件名为labels PS：理解为文件夹里的每个图片有一个序号，还有一个类别号
%生成随机数列
if (Flag_Use_Random_Index==1)
    disp('***使用随机抽取')
    Random_Q=randperm(max(labels)); % 生成102个随机排列的数组
    Random_Index=Random_Q(1:Flower_Num); % 从之中 截取 前 Flower_Num 个数字 组成新的数组，用于随机抽取 
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%以下为训练部分
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Original_Files = dir(fullfile([File_Fofer_Path,Original_Set],'*.jpg')); %读取Flower_Numflower目录地址
W=[];
disp('训练矩阵初始化完成 ')
% 循环得到图像的训练集
fprintf('处理训练集图片种类数:');
for flower_index=1:Flower_Num
    if (Flag_Use_Random_Index==1) % 判断是否使用随机抽取
        File_Index=find(labels==Random_Index(flower_index));    % 使用随机抽取得到的类别，然后得到相应的引索值
    else
        File_Index=find(labels==flower_index);  % 不使用随机抽取，取前Flower_Num个类别，得到引索值
    end
    % 主要部分 循环读取训练图片 进行训练
    for i = 1:Train_Set_Num
        flower_img_index=File_Index(i); % 读取图片的引索值
        img = imread(strcat([File_Fofer_Path,Original_Set],Original_Files(flower_img_index).name));%读取图像
        img_gray = rgb2gray(img);%转换成灰度图
        if(i==1 && Flag_Show_Picture==1) % 判断是否显示图片
            figure;imshow(img_gray) %显示每种花的样子
        end
        [m,n] = size(img_gray); % 得到图片大小
        m1 = round(m/2);n1=round(n/2);  % 找图像中心点
        img_midle = img_gray(m1-(Picture_Cut_Size/2-1):m1+(Picture_Cut_Size/2),n1-(Picture_Cut_Size/2-1):n1+(Picture_Cut_Size/2));%截取中间图像
        img_resize = imresize(img_midle,[height,width]);%调整大小
        W(:,i,flower_index) = double(img_resize(:)); %得到训练向量
    end
    printIteration(flower_index);
end
fprintf('\n');
disp('图像训练集矩阵计算完成 ')
%% 随机森林算法 训练
fprintf('RF 训练 种类数:');
Y =[ones(1,Train_Set_Num) -1*ones(1,Train_Set_Num)];    % 训练使用的数组
for A=1:Flower_Num % 循环生成 两两对比的SVM训练结果
    for B=1:Flower_Num
        W_T(:,:,A,B)=[W(:,:,A) W(:,:,B)];   % 生成对比矩阵: W_T(:,:,A,B) ，此矩阵是4维数组，PS:可以想象成一个矩阵X(A,B),A,B 是引索，得到的是一个小的2维矩阵
        if(A~=B)    % 不能自己跟自己对比，因此除自己与自己之外的其他情况，进行RF训练
            %区分第A类和第B类 
            RF_model(A,B) = classRF_train(W_T(:,:,A,B)',Y, RF_Tree);
        end
    end
    printIteration(A);
end
W=0;    % 清空内存
W_T=0;  % 清空内存
disp('随机森林 训练完成 ')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%以下为测试部分
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
correct=0;  % 总的正确个数
correct_Temp=0; % 每个类别分别的正确个数
Test_Num_ALL=0; % 总测试图片数量
for flower_index=1:Flower_Num
    if (Flag_Use_Random_Index==1) % 判断是否使用随机抽取
        File_Index=find(labels==Random_Index(flower_index));    % 使用随机抽取得到的类别，然后得到相应的引索值
    else
        File_Index=find(labels==flower_index);  % 不使用随机抽取，取前Flower_Num个类别，得到引索值
    end
    % 这里判断是否超出图片引索值 大概看了一下 每类花的图片数量为40+ 所以训练图片最多35张
    if((length(File_Index)-Train_Set_Num)<=Test_Set_Num) % 此类别花的除训练外的图片不足设定的Test_Set_Num测试图片数量时
        Test_Set_End=length(File_Index); % 测试图片的结束引索号
        Test_Pic_Num=length(File_Index)-Train_Set_Num;  % 测试图片的总数量
    else
        Test_Set_End=Test_Set_Num+Train_Set_Num;% 测试图片的结束引索号
        Test_Pic_Num=Test_Set_Num;% 测试图片的总数量
    end
    Test_Num_ALL=Test_Num_ALL+Test_Pic_Num; % 累计总的测试图片数量
    if(Flag_Test_Detail==1) % 判断是否计算测试细节
        result_T=zeros(1,Flower_Num); % 重置为0
    end
    for i = (Train_Set_Num+1):Test_Set_End
        flower_img_index=File_Index(i);
        img = imread(strcat([File_Fofer_Path,Original_Set],Original_Files(flower_img_index).name));%读取图像
        img_gray = rgb2gray(img);
        % 添加 特征提取的部分
        [m,n] = size(img_gray); % 得到图片大小
        m1 = round(m/2);n1=round(n/2);  % 找图像中心点
        img_midle = img_gray(m1-(Picture_Cut_Size/2-1):m1+(Picture_Cut_Size/2),n1-(Picture_Cut_Size/2-1):n1+(Picture_Cut_Size/2));%截取中间图像
        img_resize = imresize(img_midle,[height,width]);%调整大小
        testy = double(img_resize(:)); %训练向量
        for B=1:Flower_Num
            if(flower_index~=B)
                result(flower_index,B)  = classRF_predict(testy',RF_model(flower_index,B));%大于零为第A类，小于零为第B类
            else
                result(flower_index,B) =0;
            end
        end
        % 判断方法就是：计算 (识别的/总数) > 我们设定的阈值比率 就认为是识别的 
        % 100准确识别的话 比率 J_Rate 应设置为 0.9 或者 0.8
        if(length(find(result(flower_index,:)==1))/(length(result(flower_index,:))-1)>J_Rate)   % 判断是否为正确识别
            correct=correct+1;  %总统计
            correct_Temp=correct_Temp+1;%每个类别统计
        end
        if(Flag_Test_Detail==1) % 判断是否计算测试细节
            result(find( result<0 ))=0; % 只保留正确识别的
            result_T=result(flower_index,:)+result_T; % 累加 可以得到每个分类器的识别情况
        end
    end
    result_Rate_T = correct_Temp/Test_Pic_Num;  % 计算 每个类别测试的识别正确率
    if (Flag_Use_Random_Index==1)
        disp(['第',num2str(Random_Index(flower_index)),'类花的 RF随机森林算法 识别率为 ',num2str(result_Rate_T*100),' %'])
    else
        disp(['第',num2str(flower_index),'类花的 RF随机森林算法 识别率为 ',num2str(result_Rate_T*100),' %'])
    end
    if(Flag_Test_Detail==1) % 判断是否计算测试细节
        disp(['细节：',num2str(result_T()/Test_Pic_Num)])
    end
    correct_Temp=0; % 
end
disp('随机森林算法 测试完成 ')
disp(' ')
%% 计算识别率
result_Rate = correct/Test_Num_ALL; % 总识别率
disp(['随机森林 算法 总识别率为: ',num2str(result_Rate*100),' %'])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
toc % 计算使用时间 截止符号 与开头的 tic 对应