clear;
close all;
clc;

for i=1:100
    if(i<10)
        img_path=['AR/AR_Test_image/' 'm-00' num2str(i) '-25.bmp'];
    elseif(i>=10&&i<=50)
        img_path=['AR/AR_Test_image/' 'm-0' num2str(i) '-25.bmp'];
    else
        img_path=['AR/AR_Test_image/' 'w-0' num2str(i) '-25.bmp'];
    end    
    img=imread(img_path);    
    img=rgb2gray(img);
    if(i==1)
        [row,col]=size(img);
        imgset=zeros(row*col,1);
    end
    img=reshape(img,[],1);
    imgset(:,i)=img;
end

eigenface=readmatrix('eigenface.txt');

for d=1:4:9  
    for i=1:100
       coeff=imgset(:,i)'*eigenface(:,1:d);    
       back_img=coeff*eigenface(:,1:d)';
       back_img=mat2gray(reshape(back_img,row,col));
%        subplot(5,10,i);
%        imshow(back_img);   
        file_name=['ans(a)-ii/',num2str(d),'_',num2str(i),'.bmp'];
        imwrite(back_img,file_name);
    end
end
