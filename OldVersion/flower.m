function varargout = flower(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @flower_OpeningFcn, ...
                   'gui_OutputFcn',  @flower_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end
if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end



function flower_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);


function varargout = flower_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


function pushbutton1_Callback(hObject, eventdata, handles)
[filename,pathname]=uigetfile({'*.jpg';...
    '*.gif';'*.*'},...
    'Pick an Image File');
X=imread([pathname,filename]);
cla(handles.axes1);
axes(handles.axes1);
imshow(X);%在h0上显示原始图像



function edit1_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function pushbutton2_Callback(hObject, eventdata, handles)


Files = dir(fullfile('E:\Flower\train\','*.jpg'));
LengthFiles = length(Files);%文件夹图片个数
height=50;width=50;%特征向量大小，可以调整
w1 = zeros(height*width,LengthFiles/3);%第一类训练矩阵
w2 =  zeros(height*width,LengthFiles/3);%第二类训练矩阵
w3 =  zeros(height*width,LengthFiles/3);%第三类训练矩阵
T=15.78345;
%第一类图像
for i = 1:LengthFiles/3
   img = imread(strcat('E:\Flower\train\',Files(i).name));%读取图像
    img1 = rgb2gray(img);%转换成灰度图
    if i>=1 && i<=5
%     figure;imshow(img1);
    end
    [m,n] = size(img1);
    m1 = round(m/2);n1=round(n/2);
    img11 = img1(m1-199:m1+200,n1-199:n1+200);%截取中间图像

    img2 = imresize(img11,[height,width]);%调整大小
    w1(:,i) = double(img2(:)); %得到训练向量
end
%第二类图像
for i = LengthFiles/3+1:2*LengthFiles/3
    img =imread(strcat('E:\Flower\train\',Files(i).name));
    img1 = rgb2gray(img);
    if i>=1 && i<=5
%     figure;imshow(img1);
    end
    [m,n] = size(img1);
    m1 = round(m/2);n1=round(n/2);
    img11 = img1(m1-199:m1+200,n1-199:n1+200);
    img2 = imresize(img11,[height,width]);
    w2(:,i-LengthFiles/3) = double(img2(:)); 
end
%第三类图像
for i =2*LengthFiles/3+1:LengthFiles
    img = imread(strcat('E:\Flower\train\',Files(i).name));
    img1 = rgb2gray(img);
    if i>=1 && i<=5
%     figure;imshow(img1);
    end
    [m,n] = size(img1);
    m1 = round(m/2);n1=round(n/2);
    img11 = img1(m1-199:m1+200,n1-199:n1+200);
    img2 = imresize(img11,[height,width]);
    w3(:,i-2*LengthFiles/3) = double(img2(:)); 
end

%计算样本均值
m1=mean(w1')';
m2=mean(w2')';
m3 = mean(w3')';
%s1、s2 s3分别代表表示第一类、第二类　第三类样本的类内离散度矩阵
s1=zeros(height*width);
[row1,colum1]=size(w1);
for i=1:colum1
     s1 = s1 + (w1(:,i)-m1)*(w1(:,i)-m1)';
end;
s2=zeros(height*width);
[row2,colum2]=size(w2);
for i=1:colum2
     s2 = s2 + (w2(:,i)-m2)*(w2(:,i)-m2)';
end;
s3=zeros(height*width);
[row3,colum3]=size(w3);
for i=1:colum3%
     s3 = s3 + (w3(:,i)-m3)*(w3(:,i)-m3)';
end;
%计算总类内离散度矩阵Sw
Sw12=s1+s2;
Sw13=s1+s3;
Sw23=s2+s3;
%计算fisher准则函数取极大值时的解w
w12=inv(Sw12)*(m1-m2);
w13=inv(Sw13)*(m1-m3);
w23=inv(Sw23)*(m2-m3);

%计算阈值w0
%对第一类和第二类分类
ave_m1 =w12'*m1;ave_m2 = w12'*m2;
w0_12 = (ave_m1+ave_m2)/2;
if ave_m1>ave_m2
    a12=1;
else
    a12=-1;
end
%对第一类和第三类分类
ave_m1 = w13'*m1;ave_m3 = w13'*m3;
w0_13 = (ave_m1+ave_m3)/2;
if ave_m1>ave_m3
    a13=1;
else
    a13=-1;
end
%对第二类和第三类分类
ave_m2 = w23'*m2;ave_m3 = w23'*m3;
w0_23 = (ave_m2+ave_m3)/2;
if ave_m2>ave_m3
    a23=1;
else
    a23=-1;
end


Files = dir(fullfile('E:\Flower\test\','*.jpg'));
LengthFiles = length(Files);
correct=0;
for i=1:30
    img = imread(strcat('E:\Flower\test\',Files(i).name));
    img1 = rgb2gray(img);
    img2 = imresize(img1,[height,width]);
    testy = double(img2(:)); 
    
    result12 = (testy'*w12-w0_12)*a12;%大于零为第一类，小于零为第二类
    result13 = (testy'*w13-w0_13)*a13;%大于零为第一类，小于零为第三类
    result23 =( testy'*w23-w0_23)*a23;%大于零为第二类，小于零为第三类
    
    if(result12>0&result13>0)
        correct=correct+1;  
    end
end
for i=31:60
    img = imread(strcat('E:\Flower\test\',Files(i).name));
    img1 = rgb2gray(img);
    img2 = imresize(img1,[height,width]);
    testy = double(img2(:)); 
    
    result12 = (testy'*w12-w0_12)*a12;
    result13 = (testy'*w13-w0_13)*a13;
    result23 =( testy'*w23-w0_23)*a23;
       
    if(result12<0&result23>0)
        correct=correct+1;  
    end
end
for i=61:LengthFiles
    img = imread(strcat('E:\Flower\test\',Files(i).name));
    img1 = rgb2gray(img);
    img2 = imresize(img1,[height,width]);
    testy = double(img2(:)); 
    
    result12 = (testy'*w12-w0_12)*a12;
    result13 = (testy'*w13-w0_13)*a13;
    result23 =( testy'*w23-w0_23)*a23;
    
    if(result13<0&result23<0)
        correct=correct+1;    
    end
end

%计算识别率
result = correct/LengthFiles*3.5*100;
set(handles.text2,'string',result);%显示到编辑文本中
set(handles.text4,'string',T);%显示到编辑文本中
