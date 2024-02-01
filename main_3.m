%尺度3
%%  清空环境变量
for da=1:1:4
    for xunhuan=1:1:5
    warning off             % 关闭报警信息
    close all               % 关闭开启的图窗
%     clear                   % 清空变量
%     clc                     % 清空命令行
    
    %%  导入数据
    load CAN_A.mat;
% load CAN_B.mat;
% load A+B.mat;
% load AB.mat;
% load BA.mat;
% load BB.mat;
% load BAB.mat;
% load a.mat;
% for da=1:1:4
%     for xunhuan=1:1:5
%         load CAN_A.mat;
        if da~=1
                if da==2
                    snr=6;
                end
                if da==3
                    snr=9;
                end
                if da==4
                    snr=12;
                end   
            for i=1:1:length(res)         
                res_1=res(i,1:280);
                y1(i,:)=awgn(res_1,snr,'measured','dB');
            end
            res=[y1 res(:,281)];
        end
    
    %         res=[y1 res(:,281)];
    %     end
    %     
    %         for i=1:1:length(res)
    %             snr=12;
    %             res_1=res(i,1:280);
    %             y1(i,:)=awgn(res_1,snr,'measured','dB');
    %         end
    %         res=[y1 res(:,281)];
    %     end 
    %     if da==4
    %         for i=1:1:length(res)
    %             snr=24;
    %             res_1=res(i,1:280);
    %             y1(i,:)=awgn(res_1,snr,'measured','dB');
    %         end
    %         res=[y1 res(:,281)];
    %     end 
    % rng(100)
    % setdemorandstream(88888)
    %%  划分训练集和测试集
    % temp = randperm(length(res));
    % 
    % P_train = res(temp(1: 8000), 1: 280)';
    % T_train = res(temp(1: 8000), 281)';
    % M = size(P_train, 2);
    % 
    % P_train_1 = res(temp(8001: 10000), 1: 280)';
    % T_train_1 = res(temp(8001: 10000), 281)';
    % k = size(P_train_1, 2);
    % 
    % P_test = res(temp(10001: end), 1: 280)';
    % T_test = res(temp(10001: end), 281)';
    % N = size(P_test, 2);
    
    % temp = randperm(length(res));
    % 
    % P_train = res(temp(1: 1500), 1: 280)';
    % T_train = res(temp(1: 1500), 281)';
    % M = size(P_train, 2);
    % 
    % P_train_1 = res(temp(1501: 2000), 1: 280)';
    % T_train_1 = res(temp(1501: 2000), 281)';
    % k = size(P_train_1, 2);
    % 
    % P_test = res(temp(2001: end), 1: 280)';
    % T_test = res(temp(2001: end), 281)';
    % N = size(P_test, 2);
    
    % temp = randperm(length(res));
    % 
    % P_train = res(temp(1: 800), 1: 280)';
    % T_train = res(temp(1: 800), 281)';
    % M = size(P_train, 2);
    % 
    % P_train_1 = res(temp(801: 1100), 1: 280)';
    % T_train_1 = res(temp(801: 1100), 281)';
    % k = size(P_train_1, 2);
    % 
    % P_test = res(temp(1101: end), 1: 280)';
    % T_test = res(temp(1101: end), 281)';
    % N = size(P_test, 2);
    
    temp = randperm(length(res));
    
    P_train = res(temp(1: 4000), 1: 280)';
    T_train = res(temp(1: 4000), 281)';
    M = size(P_train, 2);
    
    P_train_1 = res(temp(4001: 5000), 1: 280)';
    T_train_1 = res(temp(4001: 5000), 281)';
    k = size(P_train_1, 2);
    
    P_test = res(temp(5001: end), 1: 280)';
    T_test = res(temp(5001: end), 281)';
    N = size(P_test, 2);
    
    %%  数据平铺
    %   将数据平铺成1维数据只是一种处理方式
    %   也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
    %   但是应该始终和输入层数据结构保持一致
    p_train =  double(reshape(P_train, 280, 1,1,M));
    p_test  =  double(reshape(P_test , 280, 1,1,N));
    P_train_1  =  double(reshape(P_train_1 , 280, 1,1,k));
    t_train =  categorical(T_train)';
    t_train_1 =  categorical(T_train_1)';
    t_test  =  categorical(T_test )';
    % setdemorandstream(88888)
    
    %%  构造网络结构
     lgraph=layerGraph();
     layers = [
         imageInputLayer([280, 1, 1],'Name','input1')
         convolution2dLayer([1 1], 4,'Name','con0','Stride', 1)
         batchNormalizationLayer         % 批归一化层
         reluLayer('Name','reluLayer')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         groupedConvolution2dLayer([128 1], 1,4,'Name','con1','Stride', 6,'Padding','same')  
         convolution2dLayer([1 1],8)
    %      dropoutLayer(0.2)
         batchNormalizationLayer         % 批归一化层
         reluLayer
         maxPooling2dLayer([2 1],'Name','global22','Stride',2)
         reluLayer('Name','RELU7')];                      
     lgraph=addLayers(lgraph,layers);
     layers = [
         groupedConvolution2dLayer([64 1],1,4,'Name','con11','Stride', 6,'Padding','same') 
         convolution2dLayer([1 1],8)
    %      dropoutLayer(0.2)
         batchNormalizationLayer         % 批归一化层
         reluLayer
         maxPooling2dLayer([2 1],'Name','global12','Stride',2)
         reluLayer('Name','RELU71')];                      
     lgraph=addLayers(lgraph,layers);
     layers = [
         groupedConvolution2dLayer([32 1], 1,4,'Name','con12','Stride', 6,'Padding','same') 
         convolution2dLayer([1 1],8)
    %      dropoutLayer(0.2)
         batchNormalizationLayer         % 批归一化层
         reluLayer
         maxPooling2dLayer([2 1],'Name','global21','Stride',2)
         reluLayer('Name','RELU72')];                      
     lgraph=addLayers(lgraph,layers);
     layers = [
         depthConcatenationLayer(3,'Name','depthConcat2')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         groupedConvolution2dLayer([3, 1], 1,24,'Name','con2')
         convolution2dLayer([1 1],24)
    %      dropoutLayer(0.2)
         batchNormalizationLayer
         reluLayer('Name','RELU1')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         globalAveragePooling2dLayer("Name",'globalA1')
         reluLayer
         fullyConnectedLayer(12,'Name','fc1')
    %      dropoutLayer(0.4)
         reluLayer
         fullyConnectedLayer(24,'Name','fc2')
    %      dropoutLayer(0.4)
         sigmoidLayer('Name','sigmoid1')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         multiplicationLayer(2,'Name','multiplication1')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         globalAveragePooling2dLayer('Name','globalA6')
         reluLayer('Name','RELU8')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         multiplicationLayer(2,'Name','multiplication8')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         groupedConvolution2dLayer([6, 1], 1,24,'Name','con3')
         convolution2dLayer([1 1],24)
    %      dropoutLayer(0.2)
         batchNormalizationLayer
         reluLayer('Name','RELU2')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         globalAveragePooling2dLayer("Name",'globalA2')
         reluLayer
         fullyConnectedLayer(12,'Name','fc4')
    %      dropoutLayer(0.4)
         reluLayer
         fullyConnectedLayer(24,'Name','fc5')
    %      dropoutLayer(0.4)
         sigmoidLayer('Name','sigmoid3')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         multiplicationLayer(2,'Name','multiplication3')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         globalAveragePooling2dLayer('Name','globalA7')
         reluLayer('Name','RELU9')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         multiplicationLayer(2,'Name','multiplication9')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         groupedConvolution2dLayer([9, 1], 1,24,'Name','con4')
         convolution2dLayer([1 1],24)
    %      dropoutLayer(0.2)
         batchNormalizationLayer('Name','batch3')
         reluLayer('Name','RELU3')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         globalAveragePooling2dLayer("Name",'globalA3')
         reluLayer
         fullyConnectedLayer(12,'Name','fc7')
    %      dropoutLayer(0.4)
         reluLayer
         fullyConnectedLayer(24,'Name','fc8')
    %      dropoutLayer(0.4)
         sigmoidLayer('Name','sigmoid5')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         multiplicationLayer(2,'Name','multiplication5')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         globalAveragePooling2dLayer('Name','globalA8')
         reluLayer('Name','RELU10')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         multiplicationLayer(2,'Name','multiplication10')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         groupedConvolution2dLayer([12, 1], 1,24,'Name','con8')
         convolution2dLayer([1 1],24)
    %      dropoutLayer(0.2)
         batchNormalizationLayer('Name','batch4')
         reluLayer('Name','RELU4')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         globalAveragePooling2dLayer("Name",'globalA5')
         reluLayer
         fullyConnectedLayer(12,'Name','fc9')
    %      dropoutLayer(0.4)
         reluLayer
         fullyConnectedLayer(24,'Name','fc10')
    %      dropoutLayer(0.4)
         sigmoidLayer('Name','sigmoid8')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         multiplicationLayer(2,'Name','multiplication7')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         globalAveragePooling2dLayer('Name','globalA9')
         reluLayer('Name','RELU11')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         multiplicationLayer(2,'Name','multiplication11')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         depthConcatenationLayer(4,'Name','depthConcat1')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         groupedConvolution2dLayer([1 1], 1,24,'Name','con9')
         convolution2dLayer([1 1],96)
    %      dropoutLayer(0.3)
         batchNormalizationLayer('Name','batch1')
    %      reluLayer('Name','RELU12')
         ];
     lgraph=addLayers(lgraph,layers); 
     layers = [
         additionLayer(2,'Name','addition7')
         reluLayer('Name','RELU12')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         groupedConvolution2dLayer([1 1],1,96,'Name','con6')
         convolution2dLayer([1 1],96)
    %      dropoutLayer(0.4)
         batchNormalizationLayer
         reluLayer
         groupedConvolution2dLayer([1 1],1,96,'Name','con7')
         convolution2dLayer([1 1],96)
    %      dropoutLayer(0.4)
         batchNormalizationLayer('Name','batch2')
    %      reluLayer('Name','relu40')
         ]; 
     lgraph=addLayers(lgraph,layers);
     layers = [
         additionLayer(2,'Name','addition8')
         reluLayer('Name','relu40')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         convolution2dLayer([1 1],1,'Name','con5')
    %      dropoutLayer(0.4)
         batchNormalizationLayer('Name','batchN5')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         reluLayer('Name','relu41')
         maxPooling2dLayer([1 1],'Name',' maxPooling2')
         reluLayer('Name','sigmoid42')
         ];
     lgraph=addLayers(lgraph,layers);
     layers = [
         reluLayer('Name','relu51')
         averagePooling2dLayer([1 1],'Name',' averagePooling2')
         reluLayer('Name','sigmoid41')
         ];
     lgraph=addLayers(lgraph,layers);
     layers = [
         depthConcatenationLayer(2,'Name','depthConcat3')];
     lgraph=addLayers(lgraph,layers);
     layers = [
         convolution2dLayer([3 1],1,'Name','con20','Padding','same')
    %      dropoutLayer(0.4)
         batchNormalizationLayer
         sigmoidLayer('Name','relu45')
         ];
     lgraph=addLayers(lgraph,layers);
    % layers = [
    %      averagePooling2dLayer([1 1],'Name','veragePool7')
    % %      batchNormalizationLayer
    %      reluLayer('Name','sigmoid9')];
    % lgraph=addLayers(lgraph,layers);
    % layers = [
    %      maxPooling2dLayer([1 1],'Name',' maxPooling2')
    %      reluLayer('Name','sigmoid11')];
    % lgraph=addLayers(lgraph,layers);
    layers = [
         multiplicationLayer(2,'Name','addition6')];
    lgraph=addLayers(lgraph,layers);
    % layers = [
    %      additionLayer(2,'Name','addition10')
    %      reluLayer('Name','reluLayer1')];
    % lgraph=addLayers(lgraph,layers);
    % layers = [
    %      convolution2dLayer([1 1],64,'Name','con10')
    %      sigmoidLayer('Name','sigmoid10')];
    % lgraph=addLayers(lgraph,layers);
    layers = [
         globalAveragePooling2dLayer('Name','globalA4')
         reluLayer('Name','sigmoid7')
         fullyConnectedLayer(10,'Name','fullyC')
    %      sigmoidLayer
         softmaxLayer
         classificationLayer('Name','output')];
     lgraph=addLayers(lgraph,layers);
    
     lgraph=connectLayers(lgraph,'reluLayer','con1');
     lgraph=connectLayers(lgraph,'reluLayer','con11');
     lgraph=connectLayers(lgraph,'reluLayer','con12');
     lgraph=connectLayers(lgraph,'RELU7','depthConcat2/in1');
     lgraph=connectLayers(lgraph,'RELU71','depthConcat2/in2');
     lgraph=connectLayers(lgraph,'RELU72','depthConcat2/in3'); 
     lgraph=connectLayers(lgraph,'depthConcat2','con2');
     lgraph=connectLayers(lgraph,'depthConcat2','con3');
     lgraph=connectLayers(lgraph,'depthConcat2','con4');
     lgraph=connectLayers(lgraph,'depthConcat2','con8');
     lgraph=connectLayers(lgraph,'depthConcat2','con9');
     lgraph=connectLayers(lgraph,'RELU1','globalA1');
     lgraph=connectLayers(lgraph,'RELU1','multiplication1/in1');
     lgraph=connectLayers(lgraph,'sigmoid1','multiplication1/in2');
     lgraph=connectLayers(lgraph,'multiplication1','globalA6');
     lgraph=connectLayers(lgraph,'RELU8','multiplication8/in1');
     lgraph=connectLayers(lgraph,'depthConcat2','multiplication8/in2');
     lgraph=connectLayers(lgraph,'RELU2','globalA2');
     lgraph=connectLayers(lgraph,'RELU2','multiplication3/in1');
     lgraph=connectLayers(lgraph,'sigmoid3','multiplication3/in2');
     lgraph=connectLayers(lgraph,'multiplication3','globalA7');
     lgraph=connectLayers(lgraph,'RELU9','multiplication9/in1');
     lgraph=connectLayers(lgraph,'depthConcat2','multiplication9/in2');
     lgraph=connectLayers(lgraph,'RELU3','globalA3');
     lgraph=connectLayers(lgraph,'RELU3','multiplication5/in1');
     lgraph=connectLayers(lgraph,'sigmoid5','multiplication5/in2');
     lgraph=connectLayers(lgraph,'multiplication5','globalA8');
     lgraph=connectLayers(lgraph,'RELU10','multiplication10/in1');
     lgraph=connectLayers(lgraph,'depthConcat2','multiplication10/in2');
     lgraph=connectLayers(lgraph,'RELU4','globalA5');
     lgraph=connectLayers(lgraph,'RELU4','multiplication7/in1');
     lgraph=connectLayers(lgraph,'sigmoid8','multiplication7/in2');
     lgraph=connectLayers(lgraph,'multiplication7','globalA9');
     lgraph=connectLayers(lgraph,'RELU11','multiplication11/in1');
     lgraph=connectLayers(lgraph,'depthConcat2','multiplication11/in2');
     lgraph=connectLayers(lgraph,'multiplication8','depthConcat1/in1');
     lgraph=connectLayers(lgraph,'multiplication9','depthConcat1/in2');
     lgraph=connectLayers(lgraph,'multiplication10','depthConcat1/in3');
     lgraph=connectLayers(lgraph,'multiplication11','depthConcat1/in4');
     lgraph=connectLayers(lgraph,'batch1','addition7/in1');
     lgraph=connectLayers(lgraph,'depthConcat1','addition7/in2'); 
     lgraph=connectLayers(lgraph,'addition6','con6');
     lgraph=connectLayers(lgraph,'batch2','addition8/in1');
     lgraph=connectLayers(lgraph,'addition6','addition8/in2');
     lgraph=connectLayers(lgraph,'RELU12','con5');
     lgraph=connectLayers(lgraph,'batchN5','relu41');
     lgraph=connectLayers(lgraph,'batchN5','relu51');
     lgraph=connectLayers(lgraph,'sigmoid42','depthConcat3/in1');
     lgraph=connectLayers(lgraph,'sigmoid41','depthConcat3/in2');
    lgraph=connectLayers(lgraph,'depthConcat3','con20');
    %  lgraph=connectLayers(lgraph,'dropout1','veragePool7');
    %  lgraph=connectLayers(lgraph,'dropout1','maxPooling2');
     lgraph=connectLayers(lgraph,'RELU12','addition6/in1');
     lgraph=connectLayers(lgraph,'relu45','addition6/in2');
    %   lgraph=connectLayers(lgraph,'relu40','addition10/in1');
    %  lgraph=connectLayers(lgraph,'addition6','addition10/in2');
    %   lgraph=connectLayers(lgraph,'addition6','con10');
     lgraph=connectLayers(lgraph,'relu40','globalA4');
    %%  绘制网络分析图
    analyzeNetwork(lgraph)
    
    
    %%  参数设置
    options = trainingOptions('adam', ...      % Adam 梯度下降算法
        'MaxEpochs',50, ...                  % 最大训练次数 500
        'MiniBatchSize',500, ...
        'InitialLearnRate', 0.001, ...          % 初始学习率为0.001
        'L2Regularization', 0.003, ...         % L2正则化参数
        'LearnRateSchedule', 'piecewise', ...  % 学习率下降
        'LearnRateDropFactor', 0.5, ...        % 学习率下降因子 0.1
        'ValidationFrequency',10, ... 
        'LearnRateDropPeriod', 30, ...        % 经过450次训练后 学习率为 0.001 * 0.5
        'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
        'ValidationData',{P_train_1,t_train_1}, ...         % 关闭验证
        'ExecutionEnvironment', 'gpu',...
        'Verbose', true ,...
        'VerboseFrequency',1);
    
    %%  训练模型
    
    % newLearnableLayer = fullyConnectedLayer(10, ...
    %     'Name','new_fc', ...
    %     'WeightLearnRateFactor',10, ...
    %     'BiasLearnRateFactor',10);
    %     
    % lgraph = replaceLayer(lgraph,'fullyC',newLearnableLayer);
    % newClassLayer = classificationLayer('Name','new_classoutput');
    % lgraph = replaceLayer(lgraph,'output',newClassLayer);
    
    net = trainNetwork(p_train, t_train, lgraph, options);
%     close all
    %%  绘制网络分析图
    % analyzeNetwork(lgraph)
    %%  预测模型
    t_sim1 = predict(net, p_train); 
    t_sim2 = predict(net, p_test ); 
    
    %%  反归一化
    T_sim1 = vec2ind(t_sim1');
    T_sim2 = vec2ind(t_sim2');
    % Y = tsne(t_sim2,'Algorithm','exact','Distance','cosine');
    % color1 = [28 28 28;
    %           244 164 96;
    %           255 193 193;
    %           0 245 255;
    %           255 69 0;
    %           34 139 34;
    %           0 252 0;
    %           132 112 255;
    %           255 255 0;
    %           139 101 8
    %           208 32 238
    %           181 181 181
    %           0 139 139
    %           193 255 193
    %           255 228 196];
    % figure
    % % subplot(2,2,1)
    % numGroups = length(unique(T_sim2));
    % clr = hsv(numGroups);
    % gscatter(Y(:,1),Y(:,2),T_sim2,color1/255,'o','filled')
    % [XRF,YRF,TRF,AUCRF] = perfcurve(T_test,T_sim2,1);
    % figure
    % plot(XRF,YRF)
    % plotroc(T_test,T_sim2);
    %%  性能评价
    error1 = sum((T_sim1 == T_train)) / M * 100 ;
    error2 = sum((T_sim2 == T_test )) / N * 100 ;
    [A,~] = confusionmat(T_test,T_sim2);
    for zhibiao=1:1:10
    c1_precise(zhibiao,1) = (A(zhibiao,zhibiao)/(A(1,zhibiao) + A(2,zhibiao)+A(3,zhibiao)+A(4,zhibiao)+A(5,zhibiao)+A(6,zhibiao)+A(7,zhibiao)+A(8,zhibiao)+A(9,zhibiao)+A(10,zhibiao)))*100;
    c1_recall(zhibiao,1) = (A(zhibiao,zhibiao)/(A(zhibiao,1) + A(zhibiao,2)+ A(zhibiao,3)+ A(zhibiao,4)+ A(zhibiao,5)+ A(zhibiao,6)+ A(zhibiao,7)+ A(zhibiao,8)+ A(zhibiao,9)+ A(zhibiao,10)))*100;
    c1_F1(zhibiao,1) = 2 * c1_precise(zhibiao,1) * c1_recall(zhibiao,1)/(c1_precise(zhibiao,1) + c1_recall(zhibiao,1));
    end
    a_precise=sum(c1_precise)/10;
    a_recall=sum(c1_recall)/10;
    a_F1=sum(c1_F1)/10;
    if da==1
    a_jieguo_n(xunhuan,1)=error2;
    a_jieguo_n(xunhuan,2)=a_precise;
    a_jieguo_n(xunhuan,3)=a_recall;
    a_jieguo_n(xunhuan,4)=a_F1;
    end
    if da==2
    a_jieguo_6(xunhuan,1)=error2;
    a_jieguo_6(xunhuan,2)=a_precise;
    a_jieguo_6(xunhuan,3)=a_recall;
    a_jieguo_6(xunhuan,4)=a_F1;
    end
    if da==3
    a_jieguo_12(xunhuan,1)=error2;
    a_jieguo_12(xunhuan,2)=a_precise;
    a_jieguo_12(xunhuan,3)=a_recall;
    a_jieguo_12(xunhuan,4)=a_F1;
    end
    if da==4
    a_jieguo_24(xunhuan,1)=error2;
    a_jieguo_24(xunhuan,2)=a_precise;
    a_jieguo_24(xunhuan,3)=a_recall;
    a_jieguo_24(xunhuan,4)=a_F1;
    end
    clearvars -except da xunhuan a_jieguo_n a_jieguo_6 a_jieguo_12 a_jieguo_24
%     clearvars -except xunhuan
%     clearvars -except a_jieguo_n
%     clearvars -except a_jieguo_6
%     clearvars -except a_jieguo_12
%     clearvars -except a_jieguo_24
    end
    clear xunhuan
end
a_jieguo=[a_jieguo_n a_jieguo_6 a_jieguo_12 a_jieguo_24];
for jisuan=1:1:16
    ai=a_jieguo(:,jisuan);
    ai_mean=sum(ai)/5;
    a_jieguo(7,jisuan)=ai_mean;
end
dlmwrite('MDSCA.txt', a_jieguo, ' ');
% clear xunhuan;
%%  数据排序
% [T_train, index_1] = sort(T_train);
% [T_test , index_2] = sort(T_test );
% 
% T_sim1 = T_sim1(index_1);
% T_sim2 = T_sim2(index_2);

%%  绘图
% figure
% plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
% legend('真实值', '预测值')
% xlabel('预测样本')
% ylabel('预测结果')
% string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
% title(string)
% xlim([1, M])
% grid
% 
% figure
% plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
% legend('真实值', '预测值')
% xlabel('预测样本')
% ylabel('预测结果')
% string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
% title(string)
% xlim([1, N])
% grid

%%  混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
    
figure
cm = confusionchart(T_test, T_sim2);
% cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.XLabel='The predicted fault';
cm.YLabel='The actual fault';
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 12)
set(gca,'position',[0.1,0.12,0.8,0.8])
set(gcf,'Units','centimeters','Position',[7,4,15,12])
% xlabel('The predicted fault','FontName','Times New Roman','FontSize',12);
% ylabel('The actual fault','FontName','Times New Roman','FontSize',12);

% figure
% plotroc(T_test,T_sim2);

