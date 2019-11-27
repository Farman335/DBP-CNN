clear all;
clc;
close all;
PSSM_Seg1_DCT=[];
PSSM_Seg2_DCT=[];
PSSM_Seg3_DCT=[];
PSSM_Seg4_DCT=[];
feature_PSSM_DCT_1=[]; 
feature_PSSM_DCT_2=[]; 
feature_PSSM_DCT_3=[]; 
feature_PSSM_DCT_4=[]; 

for a=1:14189
    a
    arr=[];
    P=csvread(['pssm' num2str(a),'.xls']);
    [L,d]=size(P);
    
    fL=round(L/4);  %%Divide sequence length into 3-segment
    
     %e=1:(1:fL)
     PA=P(1:fL,:);
     arr(:,:)=PA(:,1:20);
     S=dct2(arr);%matlab function
     U=S(1:10,1:10);
	 feature_PSSM_DCT_1(a,:)=U(:);
    % BigramPSSM_Seg1=[BigramPSSM_Seg1; lead_Bigram_PSSM(arr)];
     %PSSM_Seg1_DCT=[PSSM_Seg1_DCT; feature_PSSM_DCT_1];
    
    
    arr1=[];
    %f=fL+1:2*fL
    PB=P(fL+1:2*fL,:);
    arr1(:,:)=PB(:,1:20);
    S=dct2(arr1);%matlab function
     U=S(1:10,1:10);
	 feature_PSSM_DCT_2(a,:)=U(:);
    %BigramPSSM_Seg2=[BigramPSSM_Seg2; lead_Bigram_PSSM(arr1)];
    %PSSM_Seg2_DCT=[PSSM_Seg2_DCT; feature_PSSM_DCT_2];
    
    
    arr2=[];
    %g=2*fL+1:3*fL
    PC=P(2*fL+1:3*fL,:);
    arr2(:,:)=PC(:,1:20);
    S=dct2(arr2);%matlab function
     U=S(1:10,1:10);
	 feature_PSSM_DCT_3(a,:)=U(:);
     %PSSM_Seg3_DCT=[PSSM_Seg3_DCT; feature_PSSM_DCT_3];
    %BigramPSSM_Seg3=[BigramPSSM_Seg3; lead_Bigram_PSSM(arr2)];
    
    arr3=[];
    %h=4*fL+1:L
    PD=P(3*fL+1:L,:);
    arr3(:,:)=PD(:,1:20);
    S=dct2(arr3);%matlab function
     U=S(1:10,1:10);
	 feature_PSSM_DCT_4(a,:)=U(:);
     %PSSM_Seg4_DCT=[PSSM_Seg4_DCT; feature_PSSM_DCT_4];
    
end
PSSM_4_Seg_400_DCT_1_14189=[feature_PSSM_DCT_1 feature_PSSM_DCT_2 feature_PSSM_DCT_3 feature_PSSM_DCT_4];
%save Lead_BiPSSM_SDNA_Ind_166_Org


