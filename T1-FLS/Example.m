clc;clear;
close all;
%% seed selection
seed = 0;
rng(seed)
%% dataset location
dataset_loc = "/home/ata/AI2S/Ata/Fuzzy/dataset" % please change the dataset loc to your directory for the dataset folder
current_path = pwd;
%% housing data

cd(dataset_loc)
cd("UCI_datasets")
load("BH.mat");
cd(current_path);

data = [x y];

mbs = 64;
learnRate = 0.01;
number_of_epoch = 100;
%% power plant data

% cd(dataset_loc)
% cd("UCI_datasets")
% load("CCPP.mat");
% cd(current_path);
% 
% data = [x y];
% 
% mbs = 512; %mini batch size
% learnRate = 0.01;
% number_of_epoch = 100;
%% enb data

% cd(dataset_loc)
% cd("UCI_datasets")
% load("enb.mat");
% cd(current_path);
% 
% data = [x y];
% 
% mbs = 64; %mini batch size
% learnRate = 0.01;
% number_of_epoch = 100;
%% Other data

% data = [x y];
% 
% mbs = ~~;
% learnRate = ~~;
% number_of_epoch = ~~;
%% configuration of T1-FLS
number_of_rules = 5;
input_membership_type = "gaussmf"; %only supports gauss for now

% output_membership_type = "singleton";
output_membership_type = "linear";

tnorm = "product";
%% Adam parameters
gradDecay = 0.9;
sqGradDecay = 0.999;
averageGrad = [];
averageSqGrad = [];
%% plotting frequency
plotFrequency = 10;
%% dataset seperation proportions
fracTrain = 0.7;
fracTest = 0.3;
%%

number_inputs = size(x,2);
number_outputs = size(y,2);

%% Normalization upfront
%
[xn,input_mean,input_std] = zscore_norm(x);
[yn,output_mean,output_std] = zscore_norm(y);
    
data = [xn yn];

%% seperating data

data_size = height(data);
training_num = round(data_size*fracTrain);
test_num = data_size - (training_num);

idx = randperm(data_size);

Training_temp = data(idx(1:training_num),:);
Testing_temp = data(idx(training_num+1:end),:);

%training data
Train.inputs = reshape(Training_temp(:,1:number_inputs)', [1, number_inputs, training_num]); % traspose come from the working mechanism of the reshape, so it is a must
Train.outputs = reshape(Training_temp(:,(number_inputs+1:end))', [1, number_outputs, training_num]);

Train.inputs = dlarray(Train.inputs);
Train.outputs = dlarray(Train.outputs);

%testing data
Test.inputs = reshape(Testing_temp(:,1:number_inputs)', [1, number_inputs, test_num]);
Test.outputs = reshape(Testing_temp(:,(number_inputs+1:end))', [1, number_outputs, test_num]);


%% initializing

Learnable_parameters = initialize_Glorot_Kmeans(Train.inputs, Train.outputs, number_of_rules ,output_membership_type);
prev_learnable_parameters = Learnable_parameters;

%% seed reset
rng(seed)

%% reshaping for plotting
yTrue_train = reshape(Train.outputs,[number_outputs, training_num]);
yTrue_test = reshape(Test.outputs,[number_outputs, test_num]);
%% Training loop

number_of_iter_per_epoch = floorDiv(training_num, mbs);

number_of_iter = number_of_epoch * number_of_iter_per_epoch;
global_iteration = 1;

for epoch = 1: number_of_epoch

    [batch_inputs, batch_targets] = create_mini_batch(Train.inputs, Train.outputs, training_num);


    for iter = 1:number_of_iter_per_epoch

        [mini_batch_inputs, targets] = call_batch(batch_inputs, batch_targets,iter,mbs);

        %calculating loss and gradient
        [loss, gradients, yPred_train] = dlfeval(@fismodelLoss, mini_batch_inputs, number_inputs, targets,number_outputs, number_of_rules, mbs, Learnable_parameters, output_membership_type,tnorm);

        % updating parameters
        [Learnable_parameters, averageGrad, averageSqGrad] = adamupdate(Learnable_parameters, gradients, averageGrad, averageSqGrad,...
            iter, learnRate, gradDecay, sqGradDecay);


        global_iteration = global_iteration + 1;

    end

    %testing and plotting
    yPred_test = fismodel(Test.inputs, number_of_rules, number_inputs,number_outputs,length(Test.inputs), Learnable_parameters, output_membership_type,tnorm);
    yPred_test = reshape(yPred_test, [number_outputs, size(Test.inputs,3)]);

    plotter(epoch,plotFrequency,loss,yTrue_test, yPred_test);


end
%% Inference
yPred_train = fismodel(Train.inputs, number_of_rules, number_inputs,number_outputs,length(Train.inputs), Learnable_parameters, output_membership_type,tnorm);
yPred_test = fismodel(Test.inputs, number_of_rules, number_inputs,number_outputs,length(Test.inputs), Learnable_parameters, output_membership_type,tnorm);



yPred_train = reshape(yPred_train, [number_outputs, size(Train.inputs,3)]);
yPred_test = reshape(yPred_test, [number_outputs, size(Test.inputs,3)]);

train_RMSE = rmse(yPred_train', yTrue_train')
test_RMSE = rmse(yPred_test', yTrue_test')


%%
function [X0, targets]  = create_mini_batch(X, yTrue, minibatch_size)

shuffle_idx = randperm(size(X, 3), minibatch_size);

X0 = X(:, :, shuffle_idx);
targets = yTrue(:, :, shuffle_idx);

if canUseGPU %checking if there is a GPU available
    X0 = gpuArray(X0);
    targets = gpuArray(targets);
end

end


%%
function [mini_batch_inputs, targets] = call_batch(batch_inputs, batch_targets,iter,mbs)

mini_batch_inputs = batch_inputs(:, :, ((iter-1)*mbs)+1:(iter*mbs));
targets = batch_targets(:, :, ((iter-1)*mbs)+1:(iter*mbs));


end