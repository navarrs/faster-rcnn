%% add video path 
video_path = '/home/inavarro/Desktop/Workspace/ml/TensorFlow/keras-frcnn/images/wheelchair_videos/'
%% Add functions to path 
% addpath('video_functions')
% object = setup_system_objects('wheelchair_video_1.MP4');

video_freader = vision.VideoFileReader(fullfile(video_path, 'wheelchair_video_1.MP4'));
video_player = vision.VideoPlayer;

tic
while ~isDone(video_freader)
   frame = step(video_freader);
   view_frame = imresize(frame, 0.5);
   step(video_player,view_frame);
end
time_video = tic; 