clear;
close all;
clc;

img_list=dir('AR/AR_Train_image/*.bmp');
img_count=length(img_list);
img_path=[img_list(1).folder,'\',img_list(1).name];

for i=1:img_count
    img_path=[img_list(i).folder,'\',img_list(i).name];
    img=imread(img_path);
    img=rgb2gray(img);
    if(i==1)
        [row,col]=size(img);
        trainset=zeros(row*col,img_count);
    end    
    img=reshape(img,[],1); %將圖片矩陣展開成向量
    trainset(:,i)=img;
end

%m=mean(trainset,2);

[coeff,score,latent]=pca(trainset);

writematrix(score(:,1:9),'eigenface.txt');
for i=1:9
    eigenface=reshape(score(:,i),row,col);
    eigenface=mat2gray(eigenface);
    file_name=['ans(a)-i/' num2str(i) '.bmp'];
    imwrite(eigenface,file_name);
%     subplot(3,3,i);
%     imshow(eigenface);
end

%手刻
% m=mean(trainset,2);
% for i=1:img_count
%     trainset(:,i)=trainset(:,i)-m;
% end
% 
% c=cov(trainset);
% [eigvec,eigval]=eig(c);
% for i=0:8
%     eigenfaces=trainset*eigvec(:,1300-i);
%     temp=reshape(eigenfaces,row,col);
%     temp=mat2gray(temp);
%     subplot(3,3,i+1);
%     %file_name=['test_output/' num2str(i+1) '.bmp'];
%     %imwrite(temp,file_name);
%     imshow(temp);
% end
