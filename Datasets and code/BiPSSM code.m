clc;
clear all;
close all;

BigramPSSM_feature=[];
for i=1:5
    i
    arr=[];
    newDataPSSM=csvread(['pssm' num2str(i),'.xls']);
    arr(:,:)=newDataPSSM(:,1:20);
    BigramPSSM_feature=[BigramPSSM_feature; Bigram_PSSM(arr)];
end
BiPSSM_10001_14189=BigramPSSM_feature;
%save BiPSSM_SDNA_Ind_166_Org
