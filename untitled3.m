clc
clear all 
close all


matlabpath='C:\Users\moh\Desktop\Fruits Classification'
data=fullfile(matlabpath,'Data')

train=imageDatastore(data,'IncludeSubfolders',true,'LabelSource','foldernames');

count=train.countEachLabel;

%% load network

net=alexnet;

layers=[imageInputLayer([100 100 3])

net(2:end-3)
fullyConnectedLayer(4)
softmaxLayer
classificationLayer()
]

% % % training 
opt=trainingOptions('sgdm','Maxepoch',4,'InitialLearnRate',0.0001)

training=trainNetwork(train,layers,opt);

% % % testing 
a=imread('4.jpg')
out=classify(training,a);
figure,imshow(a)
title(string(out))



