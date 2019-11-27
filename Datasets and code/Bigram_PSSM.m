function [Bigram_features]= Bigram_PSSM(PSSM_matrix)

BigramPSSM=[];

b=size(PSSM_matrix,1);
for m=1:20
for j=1:20
    temp=0;
for i=m:b-1
    Prob_N=PSSM_matrix(i, j);
    Prob_M=PSSM_matrix(i+1, j);
    C_Prob=Prob_N*Prob_M;
    temp=temp+C_Prob;
end
    temp_b=temp/b;
    BigramPSSM=[BigramPSSM temp_b];
%     BigramPSSM(m,j)=temp_b;
end
end
    Bigram_features=BigramPSSM;
end

