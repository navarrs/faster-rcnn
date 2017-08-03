
%% Load fast RCNN workspace 
% Specify the name of the test 
test_folder = 'nda_nonmot_test1'; 

% Uncomment the following line if you wish to work with previous examples.
load(fullfile(test_folder, 'workspace', 'workspace_vars.mat'));

%% Add functions to path 
addpath('video_functions')
addpath('learning_functions');
video_path = '/home/inavarro/Desktop/Workspace/ml/TensorFlow/keras-frcnn/images/wheelchair_videos/';
% object = setup_system_objects('wheelchair_video_1.MP4');

%% Display video 
video_freader = vision.VideoFileReader(fullfile(video_path, 'wheelchair_video_9.MP4'));
video_player = vision.VideoPlayer;
video_detected = vision.VideoFileWriter('detected_video_9.avi', ...
                 'FrameRate', video_freader.info.VideoFrameRate);
            
% open(video_detected);

while ~isDone(video_freader)
   % read next frame and resize it
   frame = step(video_freader);
   resized_frame = imresize(frame, 0.3);
   
   % run the detector
   [bboxes, score, label] = detect(frcnn, resized_frame);
   [score, idx] = max(score); 
   bbox = bboxes(idx, :);
   confidence = sprintf('%s: %f', label(idx), score); 
   
   % Insert bounding box detected
   detected = insertObjectAnnotation(resized_frame, 'rectangle', bbox, confidence, ...
                      'FontSize', 14, 'LineWidth', 3, ...
                      'Color', 'green', 'Textcolor', 'black', ...
                      'TextBoxOpacity', 0.7);
                 
   % show new frame
   step(video_player, detected);
   step(video_detected, detected);
   pause(0.05);
end

release(video_freader);
release(video_player);
release(video_detected);

