
%% FASTER-RCNN TO DETECT WHEELCHAIRS IN CLUTTERED ENVIRONMENTS
%  On this code I test Faster-RCNN on a wheelchair dataset using a 
%  pre-trained model from a MATLAB vehicle dataset. The objective of this
%  approach is to detect wheelchairs given a cluttered environment and 
%  analyze how well it is able to predict the prescence of a wheelchair on 
%  a never-seen-before image. 

%  Further information on this example can be found in the following link:
%  Object detection using Faster-RCNN Deep Learning:
%  https://www.mathworks.com/help/vision/examples/
%        object-detection-using-faster-r-cnn-deep-learning.html

% Regarding this adaptation: 
%   - I used Training Image Labeler to define Regions of Interest (RoI). 
%   - Object of interest: wheelchairs
%   - Total dataset size: 445 images containing only wheelchairs. 
%   - Tested on a GeForce GT 750M, 4Gb, 386 cores. 

%% Creating new test / load previous test / load checkpoint
% To test a previous example, set test_folder to the name of the one of 
% interest want to work with. To train or re-train, change the test number 
% so that you don't override previous tests.
test_folder = 'test_3'; 

% Uncomment the following line if you wish to work with previous examples.
load(fullfile(test_folder, 'workspace', 'workspace_vars.mat'));

% Uncomment following lines if you want to start training from checkpoint. 
% ckp = load(fullfile(test_folder, 'checkpoints', ... 
  %    'faster_rcnn_stage_4_checkpoint__5904__2017_07_15__18_04_22.mat'));

%% Add my functions to MATLAB path  
addpath('learning_functions');
  
%% Prepare folders to store results
% Create folders to store results for new training process (unless you want
% to work with a previous one). 
mkdir(test_folder, 'checkpoints');
mkdir(test_folder, 'test_images');
mkdir(test_folder, 'graphs'); 
temp_dir = fullfile(pwd, test_folder, 'checkpoints'); 

%% Loading dataset
% Preparing dataset and Net to fine-tune. In this case I am extracting the 
% layers from a model that was previously trained with vehicles. This will 
% help reducing the computational cost of training from scratch. 
data = load(fullfile(matlabroot,'toolbox','vision','visiondata', ... 
            'fasterRCNNVehicleTrainingData.mat'));
layers = data.layers;

% Loading the Regions of Interest created using Image Labeler
wheelchair_data = load('ROI_reduced_dataset_resized_test.mat');
wheelchairs = wheelchair_data.wheelchairs; 
clearvars data wheelchair_data; 

%% Shuffle dataset table
% If dataset is ordered by category, this makes sure that the training process 
% will take all kinds of images. 
size = 425;
row = randperm(size, size);      % adjust value acording to dataset size.  
aux_wheelchair = wheelchairs;  
for r = 1:length(row)
    for i = 1:2
        aux_wheelchair{r, i} = wheelchairs{row(r), i};
    end
end
wheelchairs = aux_wheelchair;
clearvars aux_wheelchair size row r i;

%% Train and test sets. 
% Split all data into train and test sets. I chose the training dataset to
% be 70 - 80 % of the dataset. 
i = floor(0.80 * height(wheelchairs));
training_data = wheelchairs(1:i,:); 
test_data = wheelchairs(i:end,:);
clearvars i; 

%% Fine-tuning and layer extraction 
% Formula to determine size of a layer's output volume. 
% Output_Vol = [ ( Input_Vol - Filter_Size + 2 * Padding ) / Stride ] + 1

% First layer: imageInputLayer
input_layer = imageInputLayer([32 32 3], ...      % size
                 'Name', 'wheelchair_image', ...  % name
                 'DataAugmentation', 'randfliplr', ...  % data augmentation
                 'Normalization', 'zerocenter');  % normalization

% Middle Layers: if training model from scratch (which is time consuming 
% and computationally expensive) then set pre_trained = false. If using 
% pre-trained model, then pre_trained true. 
pre_trained = true;
if pre_trained   
    % Extract middle layers from existing model that you want to use: 
    middle_layers = layers(2:end-3);
else
    % Define new middle layers (the ones below are just an example): 
    filter_size = [3 3];
    num_filters = 32; 
    middle_layers = [
        convolution2dLayer(filter_size, num_filters, 'Padding', 1);
        reluLayer();
        convolution2dLayer(filter_size, num_filters, 'Padding', 1);
        reluLayer();
        maxPooling2dLayer(3, 'Stride', 2);
   ]; 
end

% Final layers:
num_class = 2;  % change this number to your number of classes
final_layers = [
    % NOTE: If the following are uncommented, make sure you change the 
    %       middle_layers of pre-trained model to:
    %       'layers(2:end-5)' instead of 'layers(2:end-3)'.
    %       or whatever suits the model. 
    
    % maxPooling2dLayer(3, 'Stride', 2);
    % fullyConnectedLayer(64)      
    % reluLayer('Name', 'ReLU');   
    
    % Last fully connected layer. 
    fullyConnectedLayer(num_class, ...        % output size 
                        'Name', 'LastFCL', ... 
                        'WeightLearnRateFactor', 2.0, ... %  
                        'BiasLearnRateFactor', 2.0)
    softmaxLayer('Name', 'SoftMax');
    classificationLayer('Name', 'Wheelchair');
];

% Create net layers 
net_layers = [
      input_layer
      middle_layers
      final_layers 
];

clearvars input_layer middle_layers final_layers layers pre_trained; 

%% Training Options
% EPOCH: number of times all of the training vectors are used once to
% update the weights. 
% BATCH: number of samples going to the propagated network. 

% Training option method accepts output functions, if needed to implement
% yours, you can add them to functions variable. The 'OutputFcn' option
% will take this variables

% This step will take ROIs to train an RPN. 
options_step1 = trainingOptions('sgdm', ...
      'MaxEpochs', 20, ...
      'Momentum', 0.9, ...
      'MiniBatchSize', 128, ...
      'LearnRateSchedule', 'piecewise', ...
      'LearnRateDropFactor', 0.1, ...
      'LearnRateDropPeriod', 5, ...
      'InitialLearnRate', 1e-5, ...
      'CheckpointPath', temp_dir); 
 
% Train Fast-RCNN based on the information obtained from previous step.  
options_step2 = trainingOptions('sgdm', ...
      'MaxEpochs', 20, ...
      'Momentum', 0.9, ...
      'MiniBatchSize', 128, ...
      'LearnRateSchedule', 'piecewise', ...
      'LearnRateDropFactor', 0.1, ...
      'LearnRateDropPeriod', 5, ...
      'InitialLearnRate', 1e-5, ...
      'CheckpointPath', temp_dir);
 
% This step will re-train RPN using weight sharing with Fast-RCNN
options_step3 = trainingOptions('sgdm', ...
       'MaxEpochs', 20, ...
       'Momentum', 0.9, ...
       'MiniBatchSize', 128, ...
       'LearnRateSchedule', 'piecewise', ...
       'LearnRateDropFactor', 0.1, ...
       'LearnRateDropPeriod', 5, ...
       'InitialLearnRate', 1e-6, ...
       'CheckpointPath', temp_dir);
   
% Re-train Fast-CNN with updated RPN. 
options_step4 = trainingOptions('sgdm', ...
       'MaxEpochs', 20, ...
       'Momentum', 0.9, ...
       'LearnRateSchedule', 'piecewise', ...
       'MiniBatchSize', 128, ...
       'LearnRateDropFactor', 0.1, ...
       'LearnRateDropPeriod', 5, ...
       'InitialLearnRate', 1e-6, ...
       'CheckpointPath', temp_dir); 
  
 options = [
       options_step1  % Training a Region Proposal Network (RPN)
       options_step2  % Training Fast-RCNN from RPN 
       options_step3  % Re-training RPN using weight sharing with Fast-RCNN
       options_step4  % Re-training Fast-RCNN using updated RPN
];

clearvars options_step1 options_step2 options_step3 options_step4;

%% Start training  - Net transfer
% +OVERLAP & -OVERLAP = area(AnB)/ area(AuB)
% Train the Object Detector, provide the method with at least the training
% data, the layers and the stage options that were defined. 

tic 
rng(0);
frcnn = trainFasterRCNNObjectDetector(training_data, net_layers, options, ...
        'NegativeOverlapRange', [0 0.4], ...
        'PositiveOverlapRange', [0.75 1], ...
        'NumStrongestRegions', Inf, ... % reduce to speed-up
        'BoxPyramidScale', 1.2);

training_time = tic;

%% Wheelchair save workspace variables
% If MATLAB closes for some reason, better save variables right after
% finishing training. 
mkdir(test_folder, 'workspace')
save(fullfile(pwd, test_folder, 'workspace', 'workspace_vars.mat'), ... 
                'net_layers', 'training_data', 'test_data', 'wheelchairs', ...
                'options', 'frcnn', 'training_time');    
    
%% Evaluate images 
% Using test data images to see how well it was trained. 
for i = 11:30
    % Read image path 
    image = imread(test_data.imageFilename{i});
    gnd_truth = insertShape(image, 'Rectangle', test_data.wheelchair{i}, ...
                        'LineWidth', 3, 'Color', 'red');
    
    % Calculate box, score and labels detected
    [bboxes, score, label] = detect(frcnn, image);
    
    % Only go with the highest score detected
    [score, idx] = max(score);
    bbox = bboxes(idx, :);
    confidence = sprintf('%s: %f)', label(idx), score);
    
    % Insert bounding box detected
    detected = insertObjectAnnotation(gnd_truth, 'rectangle', bbox, confidence, ...
                      'FontSize', 14, 'LineWidth', 3, ...
                      'Color', 'green', 'Textcolor', 'black', ...
                      'TextBoxOpacity', 0.7);
         
    % Add feature map to image to observe activations on the Region
    % proposal Network, right before the classification layer. 
    % Using the BoundingBoxRegresorLayer to see what features were extracted
    feat_map = activations(frcnn.Network, image, ...
                           frcnn.Network.RegressionLayers(2, 1), 3); 
    size(feat_map);
    wheelchair_map = feat_map(:, :, 1);
    [height, width, ~] = size(image);
    wheelchair_map = imresize(wheelchair_map, [height, width]);
    
    % Interpolate values
    interp_map = interpolate(wheelchair_map); 
    
    % Visualize the feature map superimposed on the test images
    feat_map1 = imfuse(image, wheelchair_map); 
    feat_map2 = imfuse(image, interp_map); 
    
    % Plot and save images
    title_detec = sprintf('Prediction: %3f', score); 
    plot_save(image, 'Original', detected, title_detec, ...
              wheelchair_map, 'Activations', interp_map, 'Filtered', ...
              feat_map1, 'Feature Map', feat_map2, 'Filtered Map', ...
              'test_img_', i, test_folder); 
end

%% Test with random images
% We'll take random images, just to see how the detector works 
for i = 1:10
    % Read image path 
    image = imread(fullfile(pwd, 'random_images', ... 
            strcat(sprintf('ri_%s', num2str(i)), '.jpg')));

    % Calculate box, score and labels detected
    [bboxes, score, label] = detect(frcnn, image);

    % Only go with the highest score detected
    [score, idx] = max(score);
    bbox = bboxes(idx, :);
    confidence = sprintf('%s: %f)', label(idx), score);
    
    % Insert bounding box detected
    detected = insertObjectAnnotation(image, 'rectangle', bboxes, confidence, ...
               'FontSize', 14, 'LineWidth', 3, ...
               'Color', 'green', 'Textcolor', 'black', ...
               'TextBoxOpacity', 0.7);
         
    % Add feature map to image to observe activations on the Region
    % proposal Network, right before the classification layer. 
    % Using the BoundingBoxRegresorLayer to see what features it extracted
    feat_map = activations(frcnn.Network, ...
               image, frcnn.Network.RegressionLayers(2, 1), 3); 
    size(feat_map);
    image_map = feat_map(:, :, 1);
    [height, width, ~] = size(image);
    image_map = imresize(image_map, [height, width]);
    
    % Interpolate activations
    interp_map = interpolate(image_map); 
    
    % Visualize the feature map superimposed on the test images
    feat_map1 = imfuse(image, image_map, 'blend'); 
    feat_map2 = imfuse(image, interp_map, 'checkerboard'); 

    % Plot and save images
    title_detec = sprintf('Prediction: %3f', score); 
    plot_save(image, 'Original', detected, title_detec, ... 
              image_map, 'Activations', interp_map, 'Interpolation', ...
              feat_map1, 'Feature Map', feat_map2, 'Feature Map (interp)', ...
              'rand_img_', i, test_folder); 
end

%% Clear memory
clearvars confidence detected feat_map feat_map1 feat_map2 title_detec ...
          gnd_truth height i idx image interp_map wheelchair_map image_map 

%% Evaluate predicitons
% Run detector on each image in the test set and collect results.
% Define path to store graphs
graphs_path = strcat(pwd, '/', test_folder, '/graphs');

%% Evaluate training data
results_struct = eval_pred(training_data, frcnn); 
% Store results in a table
results_train = struct2table(results_struct);
% Extract expected bounding box locations from train data.
expected_results = training_data(:, 2:end);
% Evaluate the object detector using Average Precision metric.
[av, recall, precision] = ...
    evaluateDetectionPrecision(results_train, expected_results);
% Plot precision / recall curve on training data 
prec_recall(precision, recall, av, 'train', graphs_path); 
% Get F1-score / iteration 
f1_score(precision, recall, 'train', graphs_path); 

%% Clear memory
clearvars av recall precision ans results_struct

%% Evaluate test data
results_struct = eval_pred(test_data, frcnn); 
% Store results in a table
results_test = struct2table(results_struct);
% Extract expected bounding box locations from test data.
expected_results = test_data(:, 2:end);
% Evaluate the object detector using Average Precision metric.
[av, recall, precision] = ... 
    evaluateDetectionPrecision(results_test, expected_results);
% Plot precision / recall curve on test data
prec_recall(precision, recall, av, 'test', graphs_path);                                                     
% Get F1-score / iteration 
f1_score(precision, recall, 'test', graphs_path); 

%% Clear memory
clearvars av recall precision ans results_struct

%% Plot layer weights 

% Plot convolutional layer weights
plotLayerWeights(frcnn.Network.Layers(2, 1).Weights, ...
                 200, 'Convolution 2D Layer Weights', ...
                 strcat(graphs_path, '/Weights_conv2D')); 

% Plot fully connected layer 1 weights             
plotLayerWeights(frcnn.Network.Layers(7, 1).Weights, ...
                 200, 'FC Layer 1 Weights', ...
                 strcat(graphs_path, '/fcl_1'));
             
% Plot fully connected layer 2 weights             
plotLayerWeights(frcnn.Network.Layers(9, 1).Weights, ...
                 200, 'FC Layer 2 Weights', ...
                 strcat(graphs_path, '/fcl_2'));            

%% Wheelchair save workspace variables
save(fullfile(pwd, test_folder, 'workspace', 'workspace_vars.mat'), ... 
                'results_train', 'results_test');