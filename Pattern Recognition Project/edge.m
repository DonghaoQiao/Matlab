%DONGHAOQIAO Final Project
%show DWT images
clear;close all;clc;

I=imread('./training/dog/golden_retriever_17.png');
J=imresize(I,[64,64]);
J=rgb2gray(J); %convert RGB image to grayscale
[~,cH,cV,~]=dwt2(J,'Haar');
nbcol=size(colormap(gray),1);
cH1=wcodemat(cH,nbcol);
cV1=wcodemat(cV,nbcol);
edges=cH1+cV1;
nw=32*32;
dcData=reshape(edges,nw,1);

figure(1);
subplot(241);imshow(J);title('Grayscale Image');
subplot(242);image(cH1);title('Horizontal Image');
subplot(243);image(cV1);title('Vertical Image');
subplot(244);image(edges);title('Edge Image');


I=imread('./training/cat/Abyssinian_1.jpg');
J=imresize(I,[64,64]);
J=rgb2gray(J); %convert RGB image to grayscale
[~,cH,cV,~]=dwt2(J,'Haar');
nbcol=size(colormap(gray),1);
cH1=wcodemat(cH,nbcol);
cV1=wcodemat(cV,nbcol);
edges=cH1+cV1;
nw=32*32; 
dcData=reshape(edges,nw,1);

figure(1);
subplot(245);imshow(J);title('Grayscale Image');
subplot(246);image(cH1);title('Horizontal Image');
subplot(247);image(cV1);title('Vertical Image');
subplot(248);image(edges);title('Edge Image');



