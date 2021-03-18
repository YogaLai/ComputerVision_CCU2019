clear all;
img=imread('Shiba_1.jpg');
% img=imread('Shiba2.jpg');
maskout=kGaussian_color_EM(img,5);
imshow(maskout);
title('Use 5 Gaussion mixtures models');

function [maskOut]=kGaussian_color_EM(imageFile,k)
    %% this function uses EM algorithm and gaussian distribution function to do
    img=double(imageFile);
    [M,N,P]=size(img);
    n=M*N;
    imgR=img(:,:,1); 
    imgG=img(:,:,2);
    imgB=img(:,:,3);

    imgR=imgR/255;
    imgG=imgG/255;
    imgB=imgB/255;

    %% assign vectors into the matrix raw
    raw=zeros(n,3);
    raw(:,1)=imgR(:);
       raw(:,2)=imgG(:);
    raw(:,3)=imgB(:);

    %% get assignment matrix p, which is also memebership probability here; u
    %% is vector of the estimated means of Gaussian function, v is the vector ot the estimated SD 
    [p,u,v]=myEM(raw,k);
    imgRe=zeros(n,3);
    kColor=jet(k);
    kColor=u(:,1:3);
    imgRe=p*kColor;
    maskOut=zeros(M,N,3);
    for ii=1:3
        maskOut(:,:,ii)=reshape(imgRe(:,ii),[M,N]);
    end
end
