%DONGHAOQIAO Final Project
%Support Vector Machine
function [Group_svm]=svm(dogs,cats,U)
nd=length(dogs(1,:)); %677
nc=length(cats(1,:)); %938

Training = [dogs cats]';

% Label matrix for svmtrain
DOGS_train = zeros(1,nd);
CATS_train = ones(1,nc);
Train_Labels = [DOGS_train CATS_train]';

% Use fitcsvm classifier
SVMModel = fitcsvm(Training,Train_Labels,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');

% Classify the test set using predict function
TestSet='./testing';
[Test_wave,~]=wavelet(TestSet); %wavelet transformation
TestMat=U'*Test_wave; %SVD projection
Group_svm = predict(SVMModel,TestMat');
end
