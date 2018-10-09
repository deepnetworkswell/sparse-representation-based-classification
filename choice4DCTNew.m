


function choice4DCTNew()
%% SRC face recognition (sparse representation-based classification)

clear all;
clc;

subjectCounts=40;  %number of subjects

[Dictionary_A,testFaces]=featureSet();   %dictionary using downsampled faces

%%for use in LCKSVD
% training_feats=Dictionary_A;
% testing_feats=testFaces;
% save('.\downsample\featurevectors.mat','training_feats','testing_feats');
%*********
recNumber=0;      %number of correct recoginition
for subject=1:subjectCounts;
    for picNo=6:10
        testFaceV=testFaces(:,(subject-1)*5+picNo-5);
        x=BP(Dictionary_A,testFaceV);   %solve the L1 optimization problem
        for i=1:subjectCounts;
        x_i=zeros(200,1);
        x_i((i*5-4) : (i*5))=x((i*5-4) : i*5);
        residual_i=norm(testFaceV-Dictionary_A*x_i);  %compute residual for i_th subject
        residualV(i)=residual_i;
        % %reconstruct face
        % colormap gray;
        % imagesc((reshape((Dictionary_A*x_i),14, 17))');
        end
        [min_residual,identity]=min(residualV);
        if identity == subject
            recNumber=recNumber+1;
        end
    end
end

recRate=recNumber/200;
display(strcat('Recognition Rate=',num2str(recRate)));

% % % Plot residual error                                                 
% figure
% bar(residualV)
% xlabel('subject')
% ylabel('residual error')
% title('Experiment')

end


function [Dictionary_A,testFaces]=featureSet()
[faceDataset]=importFaces();


%% extracting DCT feature
subjectCounts=40;

for i=1:subjectCounts
    for j=1:10
        face=faceDataset(:,:,(i-1)*10+j);      
        faceDct=dct2(face);
        blocksize=2;         %chose the dimension of the top-left block coeffs of dct transform of face
        tlblock=faceDct(1:blocksize,1:blocksize);
        dicRows=blocksize*blocksize;
        dctFeatureV=reshape(tlblock,dicRows,1);
        dctFeatureV=dctFeatureV/norm(dctFeatureV);
        
        dctFeatureDataset(:,(i-1)*10+j)=dctFeatureV;    %insert the face Dct feature into one column        
    end
end

for i=1:subjectCounts
    for j=1:5
        Dictionary_A(:,(i-1)*5+j)= dctFeatureDataset(:,(i-1)*10+j);  %build dictionary
        testFaces(:,(i-1)*5+j)=dctFeatureDataset(:,(i-1)*10+j+5);    %build test dataset
    end
end

end

function [faceDataset]=importFaces()
%% Import faces
%Load all the faces
subjectCounts=40;  %number of training subjects
for i=1:subjectCounts
    for j=1:10
        folder='C:\Users\sadegh\Desktop\compr sensing course\project2\faceDataset\s';
        faceAddress=strcat(folder,num2str(i),'\',num2str(j),'.pgm');
        face=imread(faceAddress);
%         imshow(face);
        faceDataset(:,:,(i-1)*10+j)=double(face);

    end
end

end
