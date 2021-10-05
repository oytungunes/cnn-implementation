clear;
close all;
clc;
%loading data and normalization
tic;
load('final_project_dataset.mat');
pixelNum=32;
for i=1:size(testdat,1)
    aa=testdat(i,:);
    testdatShaped(:,:,:,i)=reshape(aa, [pixelNum pixelNum 3]);
end
for i=1:size(traindat,1)
    aa=traindat(i,:);
    traindatShaped(:,:,:,i)=reshape(aa, [pixelNum pixelNum 3]);
end
% preprocess test data for w-b and normalize
nbWTrainData= zeros(pixelNum,pixelNum,size(traindatShaped,4));
for i=1:size(traindatShaped,4)
    nbWTrainData(:,:,i)=rgb2gray(traindatShaped(:,:,:,i));
end
% preprocess train data for w-b and normalize
nbWTestData= zeros(pixelNum,pixelNum,size(testdatShaped,4));
for i=1:size(testdatShaped,4)
    nbWTestData(:,:,i)=rgb2gray(testdatShaped(:,:,:,i));
end
nbWTrainData=double(nbWTrainData);
nbWTestData=double(nbWTestData);
%modify train and test labels by increasing 1
trainlbl = trainlbl +1;
testlbl = testlbl + 1;

%initialize weights and kernel
ker1L=5;
ker1Num=16;
pl1=2;
numOfHid= ker1Num*((pixelNum-ker1L+1)/pl1)^2;
numOfOut=10;
kerCons1= sqrt(6/(ker1L*ker1L+900));  % how to get 4 is ker1L^2*1/pl1^2
ker1= -1*kerCons1+ (kerCons1*2)*rand(ker1L,ker1L,ker1Num);
ker1=0.05*randn(ker1L,ker1L,ker1Num);
biasKer1=-1*kerCons1+ (kerCons1*2)*rand(pixelNum-ker1L+1,pixelNum-ker1L+1,ker1Num);
biasKer1=0.05*randn(pixelNum-ker1L+1,pixelNum-ker1L+1,ker1Num);
weightConsOut= sqrt(6/(numOfHid+900)); %how to get 200
weightOut= -1*weightConsOut+ (weightConsOut*2)*rand(numOfHid,numOfOut);
biasOut=-1*weightConsOut+ (weightConsOut*2)*rand(1,numOfOut); 
lambda=1;
eta=0.0001;
alpha=0;
toc;
epochNum=5;
miniBatchNum=128;


confusiontest=zeros(numOfOut,numOfOut);

accEpoch=zeros(1,epochNum);
CostEpoch=zeros(1,epochNum);

for epochIter=1:epochNum
    eta=0.0001*(0.5^(epochIter-1));% decrease learning rate over epochs
    tic;
    picSequence = randperm(length(trainlbl));
    prevWODel= zeros(numOfHid,numOfOut);
    prevBODel=zeros(1,numOfOut);
    prevK1Del=zeros(ker1L,ker1L,ker1Num);
    prevBiasKer1Del=zeros(pixelNum-ker1L+1,pixelNum-ker1L+1,ker1Num);
    % mini-batch
    for batchIter=1:floor(length(picSequence)/miniBatchNum)
        storeWODel= zeros(numOfHid,numOfOut);
        storeBODel=zeros(1,numOfOut);
        storeK1Del=zeros(ker1L,ker1L,ker1Num);
        storeBiasKer1Del=zeros(pixelNum-ker1L+1,pixelNum-ker1L+1,ker1Num);
        for mBIter=1:miniBatchNum
            X=nbWTrainData(:,:,picSequence((batchIter-1)*miniBatchNum+mBIter));
            
            %convolutional layer
            conv1Out=zeros(pixelNum-ker1L+1,pixelNum-ker1L+1,ker1Num);
            for ker1Iter=1:ker1Num
                conv1Out(:,:,ker1Iter) = conv2(X,ker1(:,:,ker1Iter),'valid'); 
            end
            conv1OutAct=1./(1+exp(-lambda*(conv1Out-biasKer1)));    
            
            % max pooling layer
            pl1Out= zeros((pixelNum-ker1L+1)/pl1,(pixelNum-ker1L+1)/pl1,ker1Num);
            pl1OutIndex=zeros(2,(pixelNum-ker1L+1)/pl1,(pixelNum-ker1L+1)/pl1,ker1Num);
            for ker1Iter=1:ker1Num
                for i=1:(pixelNum-ker1L+1)/pl1
                    for k=1:(pixelNum-ker1L+1)/pl1
                        dumy= conv1OutAct((i-1)*pl1+1:i*pl1 ,(k-1)*pl1+1:k*pl1 ,ker1Iter);
                        [maxVal, maxInd] = max(dumy(:));
                        [Irow, Icol] = ind2sub(size(dumy),maxInd);
                        pl1Out(i,k,ker1Iter) = maxVal;
                        pl1OutIndex(1,i,k,ker1Iter)=Irow;
                        pl1OutIndex(2,i,k,ker1Iter)=Icol;
                    end
                end
            end
            %fully connected layer
            hiddenOut=pl1Out(:);
            neuralOut=transpose(hiddenOut)*weightOut-biasOut;
            % output layer
            sumOut=sum(exp(neuralOut));
            neuralOutAct= exp(neuralOut)*1/(sumOut);
            %backpropogation 
            d=zeros(1,numOfOut);
            d(1,trainlbl(picSequence((batchIter-1)*miniBatchNum+mBIter)))=1;
            gradOut= neuralOutAct - d; 
            gradHidden= weightOut*(transpose(gradOut));
            gradpl1=reshape(gradHidden, [(pixelNum-ker1L+1)/pl1 (pixelNum-ker1L+1)/pl1 ker1Num]);

            gradconv1=zeros(pixelNum-ker1L+1,pixelNum-ker1L+1,ker1Num);
            for ker1Iter=1:ker1Num
                for i=1:(pixelNum-ker1L+1)/pl1
                    for k=1:(pixelNum-ker1L+1)/pl1
                        indRow=(i-1)*pl1+pl1OutIndex(1,i,k,ker1Iter);
                        indCol=(k-1)*pl1+pl1OutIndex(2,i,k,ker1Iter);
                        gradconv1(indRow,indCol,ker1Iter)=gradpl1(i,k,ker1Iter)*conv1OutAct(indRow,indCol,ker1Iter)*(1-conv1OutAct(indRow,indCol,ker1Iter));        
                    end
                end
            end
            storeWODel=storeWODel -1*eta*(hiddenOut)*gradOut;
            weightOut =weightOut+  -1*eta*(hiddenOut)*gradOut + alpha*prevWODel;
            prevWODel=alpha*prevWODel+-1*eta*(hiddenOut)*gradOut;
            storeBODel=storeBODel -1*eta*(-1)*gradOut;
            biasOut=biasOut+ -1*eta*(-1)*gradOut;
            prevBODel=alpha*prevBODel -1*eta*(-1)*gradOut;
            
            ker1Update=zeros(ker1L,ker1L,ker1Num);
            for ker1Iter=1:ker1Num
                for i=1:ker1L
                    for k=1: ker1L
                        ind1=ker1L-i+1;
                        ind2=pixelNum-i+1;
                        ind3=k;
                        ind4=k+size(conv1Out,1)-1;
                        ker1Update(:,:,ker1Iter)=ker1Update(:,:,ker1Iter)-1*(eta)*sum(sum(X(ind1:ind2,ind3:ind4).*gradconv1(:,:,ker1Iter))); %*1/((pixelNum-ker1L+1)^2)
                    end
                end
            end
            storeK1Del=storeK1Del+ker1Update;
            storeBiasKer1Del=storeBiasKer1Del -1*eta*(-1)*gradconv1;
            aaaaa=0;
        end
        % update the kernels and weights
        ker1=ker1+(1/miniBatchNum)*storeK1Del+alpha*prevK1Del;
        prevK1Del=alpha*prevK1Del+(1/miniBatchNum)*storeK1Del;
        biasKer1=biasKer1+(1/miniBatchNum)*storeBiasKer1Del + alpha*prevBiasKer1Del;
        prevBiasKer1Del=prevBiasKer1Del++(1/miniBatchNum)*storeBiasKer1Del;
        weightOut =weightOut+  (1/miniBatchNum)*storeWODel + alpha*prevWODel;
        prevWODel=alpha*prevWODel+(1/miniBatchNum)*storeWODel;
        biasOut=biasOut+ (1/miniBatchNum)*storeBODel+ alpha*prevBODel;
        prevBODel=alpha*prevBODel + (1/miniBatchNum)*storeBODel;

    end
    toc;
    
    %test validation data
    tic;
    testError=0;
    testShuf=randperm(size(nbWTrainData,3));
    for testIter=1:1000
        X=nbWTrainData(:,:,testShuf(testIter));
        conv1Out=zeros(pixelNum-ker1L+1,pixelNum-ker1L+1,ker1Num);
        for ker1Iter=1:ker1Num
            conv1Out(:,:,ker1Iter) = conv2(X,ker1(:,:,ker1Iter),'valid'); 
        end
        conv1OutAct=1./(1+exp(-lambda*(conv1Out-biasKer1)));    
        pl1Out= zeros((pixelNum-ker1L+1)/pl1,(pixelNum-ker1L+1)/pl1,ker1Num);
        pl1OutIndex=zeros(2,(pixelNum-ker1L+1)/pl1,(pixelNum-ker1L+1)/pl1,ker1Num);
        for ker1Iter=1:ker1Num
            for i=1:(pixelNum-ker1L+1)/pl1
                for k=1:(pixelNum-ker1L+1)/pl1
                    dumy= conv1OutAct((i-1)*pl1+1:i*pl1 ,(k-1)*pl1+1:k*pl1 ,ker1Iter);
                    [maxVal, maxInd] = max(dumy(:));
                    [Irow, Icol] = ind2sub(size(dumy),maxInd);
                    pl1Out(i,k,ker1Iter) = maxVal;
                    pl1OutIndex(1,i,k,ker1Iter)=Irow;
                    pl1OutIndex(2,i,k,ker1Iter)=Icol;
                end
            end
        end
        hiddenOut=pl1Out(:);
        neuralOut=transpose(hiddenOut)*weightOut-biasOut;
        sumOut=sum(exp(neuralOut));
        neuralOutAct= exp(neuralOut)*1/(sumOut);
        d=trainlbl(testShuf(testIter));
        [mV, mI]= max(neuralOutAct);
        if (mI ~= d)
            testError=testError+1;
        end
        dvec=zeros(1,numOfOut);
        dvec(1,d)=1;
        CostEpoch(1,epochIter)=-dvec*transpose(log(neuralOutAct));

    end
    testError*(100/1000)
    accEpoch(1,epochIter)=testError*(100/1000);
    
    toc;
end

% Calculating test accuracy

testError=0;
for testIter=1:size(nbWTestData,3)
    X=nbWTestData(:,:,testIter);
    conv1Out=zeros(pixelNum-ker1L+1,pixelNum-ker1L+1,ker1Num);
    for ker1Iter=1:ker1Num
        conv1Out(:,:,ker1Iter) = conv2(X,ker1(:,:,ker1Iter),'valid'); 
    end
    conv1OutAct=1./(1+exp(-lambda*(conv1Out-biasKer1)));    
    pl1Out= zeros((pixelNum-ker1L+1)/pl1,(pixelNum-ker1L+1)/pl1,ker1Num);
    pl1OutIndex=zeros(2,(pixelNum-ker1L+1)/pl1,(pixelNum-ker1L+1)/pl1,ker1Num);
    for ker1Iter=1:ker1Num
        for i=1:(pixelNum-ker1L+1)/pl1
            for k=1:(pixelNum-ker1L+1)/pl1
                dumy= conv1OutAct((i-1)*pl1+1:i*pl1 ,(k-1)*pl1+1:k*pl1 ,ker1Iter);
                [maxVal, maxInd] = max(dumy(:));
                [Irow, Icol] = ind2sub(size(dumy),maxInd);
                pl1Out(i,k,ker1Iter) = maxVal;
                pl1OutIndex(1,i,k,ker1Iter)=Irow;
                pl1OutIndex(2,i,k,ker1Iter)=Icol;
            end
        end
    end
    hiddenOut=pl1Out(:);
    neuralOut=transpose(hiddenOut)*weightOut-biasOut;
    sumOut=sum(exp(neuralOut));
    neuralOutAct= exp(neuralOut)*1/(sumOut);
    d=testlbl(testIter);
    [mV, mI]= max(neuralOutAct);
    %confusion matrix
    confusiontest(d,mI)=confusiontest(d,mI)+1;
    if (mI ~= d)
        testError=testError+1;
    end
    
end
testerrorfinal = testError*(100/size(nbWTestData,3));
% plotting  curves
plot(CostEpoch)
xlabel 'Epochs';
ylabel 'Cross Entropy Cost';

figure
accuracyepochs=100-accEpoch;
plot(accuracyepochs)
xlabel 'Epochs';
ylabel 'Validation Accuracy ';

figure
for i=1:size(ker1,3)
    subplot(4,4,i)
    imagesc(ker1(:,:,i));
end



imagesc(confusiontest)









