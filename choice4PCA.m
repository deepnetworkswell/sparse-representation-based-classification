


function choice4PCA()
%% SRC face recognition (sparse representation-based classification)

clear all;
clc;
 subjectCounts=40;  %number of subjects

[Dictionary_A,testFaces]=featureSet();   %dictionary using downsampled faces

%%for use in LCKSVD
training_feats=Dictionary_A;
testing_feats=testFaces;
save('.\pca\featurevectors.mat','training_feats','testing_feats');
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


%% PCA
subjectCounts=40;
downsamplingRate=1;
face=faceDataset(:,:,1);   %example face
downsampledFace=imresize(face,downsamplingRate);
[s1,s2]=size(downsampledFace);   %find the size of the downsampled face
dicRows=s1*s2;  %number of rows of the dictionary


for i=1:subjectCounts
    for j=1:10
        face=faceDataset(:,:,(i-1)*10+j);
        downsampledFace=imresize(face,downsamplingRate);  %downsample faces
        faceV=reshape(downsampledFace',dicRows,1);  %reshape 2dim downsampled face into a vector
        faceV=faceV/norm(faceV);      %normalize each column
        faceDatasetV(:,(i-1)*10+j)=faceV;    %insert the face into one column
    end
end


for i=1:subjectCounts
    for j=1:5
       testFaceSet(:,(i-1)*5+j)=faceDatasetV(:,(i-1)*10+j+5);    %build test dataset
       trainFaces(:,(i-1)*5+j)= faceDatasetV(:,(i-1)*10+j);  %build dictionary
 
    end
end


for i=1:subjectCounts
    for j=1:5
       meanFace=sum(trainFaces,2)/(subjectCounts*5);
%         colormap gray;
%         imagesc((reshape(meanFace,92, 112))');
       
       trainFaces(:,(i-1)*5+j)=trainFaces(:,(i-1)*5+j)-meanFace;
%        colormap gray;
%        imagesc((reshape(trainFaces(:,(i-1)*5+j),92, 112))');
       
       testFaceSet(:,(i-1)*5+j)=testFaceSet(:,(i-1)*5+j)-meanFace;

    end
end

R=(trainFaces'*trainFaces)/(subjectCounts*5);
[eiVectorsR,eiValues]=eig(R);
eigenFaces=trainFaces*eiVectorsR;           %find eifgenvectors of XX'

        colormap gray;
        imagesc((reshape(eigenFaces(:,198),92, 112))');

for i=1:size(eigenFaces,2)
eigenFaces(:,i)=eigenFaces(:,i)/max(eiValues(:,i));    %normalization of each eigenFace
end  
eigenFaces100=eigenFaces(:,81:200);
Dictionary_A=eigenFaces100'*trainFaces;   %build dictionary using PCA by finding the projection of each image vector on the facespace 
testFaces=eigenFaces100'*testFaceSet;     %find the eigenface rep. of each test face(the projection of each test image vector on the facespace)


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
        faceDataset(:,:,(i-1)*10+j)=double(face);

    end
end

end
