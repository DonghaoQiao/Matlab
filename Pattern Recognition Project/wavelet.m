%DONGHAOQIAO Final Project
%Preprocess the figures and obtain edges

function [dcData,imgData]=wavelet(dc_folder_path)
    FolderInfo=dir(dc_folder_path); %folder contents
    allNames={FolderInfo.name};
    nw=32*32; %wavelet resolution
    index=1;
    for i=3:length(allNames)
        filename=fullfile(allNames{i}); %build full file name from parts
        filename=fullfile(dc_folder_path,filename);
        display(filename)
        %preprocess 
        I=imread(filename);
        J=imresize(I,[64,64]);
        J=rgb2gray(J); %convert RGB image to grayscale image
        [~,cH,cV,~]=dwt2(J,'Haar'); %discrete 2-D wavelet transform
        nbcol=size(colormap(gray),1);
        cod_cH1=wcodemat(cH,nbcol); %extended pseudocolor matrix scaling
        cod_cV1=wcodemat(cV,nbcol);
        cod_edge=cod_cH1+cod_cV1;
        dcData(:,index)=reshape(cod_edge,nw,1);
        imgData{index}=J;
        index=index+1;
    end
end


