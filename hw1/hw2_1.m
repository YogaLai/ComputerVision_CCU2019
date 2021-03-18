clear;
close all;
clc;

test_list=dir('AR/AR_Test_image/*.bmp');
train_list=dir('AR/AR_Train_image/*.bmp');
img_count=length(test_list);

%抓取train/test image
for i=1:img_count
    img_path=[test_list(i).folder,'\',test_list(i).name];
    img=imread(img_path);
    img=rgb2gray(img);
    if(i==1)
        [row,col]=size(img);
        trainset=zeros(row*col,img_count);
        testset=zeros(row*col,img_count);
    end    
    img=reshape(img,[],1);
    testset(:,i)=img;
    
    img_path=[train_list(i).folder,'\',train_list(i).name];
    img=imread(img_path);
    img=rgb2gray(img);
    img=reshape(img,[],1);
    trainset(:,i)=img;
end

eigenface=readmatrix('eigenface.txt');

for d=1:4:9    
    project_test=testset'*eigenface(:,1:d);
    project_train=trainset'*eigenface(:,1:d);
    %利用knnn計算error rate
    idx=knnsearch(project_test,project_train);
    error=0;
    for i=1:img_count
        if idx(i)~=i
            error=error+1;
        end
    end
    fprintf('error rate :%f\n',error/img_count);
end


