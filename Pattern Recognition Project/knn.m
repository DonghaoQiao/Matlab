%DONGHAOQIAO Final Project
%K-Nearest Neighbor
function [Group_knn]=knn(dogs,cats,U)
nd=length(dogs(1,:)); %676
nc=length(cats(1,:)); %938

Training = [dogs cats]';

% Label matrix for knn
DOGS_train = zeros(1,nd);
CATS_train = ones(1,nc);
Train_Labels = [DOGS_train CATS_train]';

% Use fitcknn classifier
KNNModel = fitcknn(Training,Train_Labels);

% Classify the test set using predict function
TestSet='./testing';
[Test_wave,~]=wavelet(TestSet); %wavelet transformation
TestMat=U'*Test_wave; %SVD projection
Group_knn = predict(KNNModel,TestMat');
end

