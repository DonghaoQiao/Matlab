%DONGHAOQIAO Final Project
%Linear Discriminent Analysis
function [Group_lda]=lda(dogs,cats,U)
nd=length(dogs(1,:)); %677
nc=length(cats(1,:)); %938
md=mean(dogs,2); %compute the average of each row
mc=mean(cats,2); %compute the average of each row
    
Sw=0; %within-class scatter matrix
for i=1:nd
    Sw=Sw+(dogs(:,i)-md)*(dogs(:,i)-md)';
end
for i=1:nc
    Sw=Sw+(cats(:,i)-mc)*(cats(:,i)-mc)';
end
    
Sb=(md-mc)*(md-mc)'; %between-class scatter matrix
[V2,D]=eig(Sb,Sw); %linear discriminant analysis(Sb*V2=Sw*V2*D)
[~,ind]=max(abs(diag(D))); %the row indices of D in which they appear
w=V2(:,ind);
w=w/norm(w,2);
vdog=w'*dogs; vcat=w'*cats;
    
if mean(vdog)>mean(vcat)
    w=-w;
    vdog=-vdog;
    vcat=-vcat;
end
sortdog=sort(vdog);
sortcat=sort(vcat);
t1=length(sortdog);
t2=1;
while sortdog(t1)>sortcat(t2)
    t1=t1-1;
    t2=t2+1;
end
threshold=(sortdog(t1)+sortcat(t2))/2;

%Test on the testing dataset:
TestSet='./testing';
[Test_wave,~]=wavelet(TestSet); %wavelet transformation
TestMat=U'*Test_wave; %SVD projection
pval=w'*TestMat; %LDA projection
Group_lda=double((pval>threshold))';
end



