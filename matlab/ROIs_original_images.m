path = ' 
%% Load label session 
load(fullfile(pwd, 'ROI_reduced_dataset_test.mat')); 
path = '/home/inavarro/Desktop/Workspace/ml/TensorFlow/keras-frcnn/csv/';

%% Shuffle data on wheelchair table
size = 445;
row = randperm(size, size);      % adjust value acording to dataset size.  
aux_wheelchair = wheelchair;  
for r = 1:length(row)
    for i = 1:2
        aux_wheelchair{r, i} = wheelchair{row(r), i};
    end
end
wheelchair = aux_wheelchair;
clearvars aux_wheelchair size row r i;

%% Train and test sets. 
% Split all data into train and test sets. I chose the training dataset to
% be 70 - 80 % of the dataset. 
i = floor(0.75 * height(wheelchair));
training_data = wheelchair(1:i,:); 
test_data = wheelchair(i:end,:);
clearvars i; 

%% Write everythiing to csv file 
writetable(wheelchair, ...
    fullfile('/home/inavarro/Desktop/Workspace/ml/TensorFlow/keras-frcnn/csv/',...
    '06_wheelchair_reduced.csv')); 

%%
writetable(training_data, ...
    fullfile(path, '03_wheelchair_training.csv')); 

writetable(test_data, ...
    fullfile(path, '03_wheelchair_test.csv')); 
