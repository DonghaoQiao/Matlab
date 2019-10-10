%DONGHAOQIAO Final Project
%Image Recognition for Dogs and Cats
clear;close all;clc;

dog_folder_path='./training/dog';
cat_folder_path='./training/cat';
[dog0,imgDog]=wavelet(dog_folder_path);
[cat0,imgCat]=wavelet(cat_folder_path);

feature=50; %1<feature<1024=32*32(wavelet resolution)
nd=length(dog0(1,:)); %676
nc=length(cat0(1,:)); %938

[U,S,V]=svd([dog0,cat0],0); %singular value decomposition of symbolic matrix
animals=S*V';
U=U(:,1:feature);
dogs=animals(1:feature,1:nd);
cats=animals(1:feature,nd+1:nd+nc);

[Group_lda]=lda(dogs,cats,U);
[Group_knn]=knn(dogs,cats,U);
[Group_svm]=svm(dogs,cats,U);
hiddenlabels=[ones(16,1);zeros(16,1)];

%Test on the testing dataset:
TestSet='./testing';
[Test_wave,imgTest]=wavelet(TestSet); %wavelet transformation

%LDA
TestNum=length(Group_lda);
errNum=sum(abs(Group_lda-hiddenlabels));
sucRate=1-errNum/TestNum;
fprintf('Number of LDA misrecognition: %f\n',errNum);
fprintf('LDA Recognition Rate: %f\n',sucRate);

k=1;
figure(1);
for i=1:TestNum
    if Group_lda(i)~=hiddenlabels(i)
        S=imgTest{i};
        subplot(3,3,k)
        imshow(uint8(S))
        title('misrecognition');
        k=k+1;
    end
end

%KNN
TestNum=length(Group_knn);
errNum=sum(abs(Group_knn-hiddenlabels));
sucRate=1-errNum/TestNum;
fprintf('Number of KNN misrecognition: %f\n',errNum);
fprintf('KNN Recognition Rate: %f\n',sucRate);

k=1;
figure(2);
for i=1:TestNum
    if Group_knn(i)~=hiddenlabels(i)
        S=imgTest{i};
        subplot(3,3,k)
        imshow(uint8(S))
        title('misrecognition');
        k=k+1;
    end
end

%SVM
TestNum=length(Group_svm);
errNum=sum(abs(Group_svm-hiddenlabels));
sucRate=1-errNum/TestNum;
fprintf('Number of SVM misrecognition: %f\n',errNum);
fprintf('SVM Recognition Rate: %f\n',sucRate);

k=1;
figure(3);
for i=1:TestNum
    if Group_svm(i)~=hiddenlabels(i)
        S=imgTest{i};
        subplot(3,3,k)
        imshow(uint8(S))
        title('misrecognition');
        k=k+1;
    end
end






