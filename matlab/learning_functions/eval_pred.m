%% Evaluate predictions 
function results = eval_pred(data, frcnn)
    results = struct([]); 
    for i = 1:size(data)
        % Read the image.
        I = imread(data.imageFilename{i});
        % Run the detector.
        [bboxes, scores, labels] = detect(frcnn, I);
        % Collect the results.
        results(i).Boxes = bboxes;
        results(i).Scores = scores;
        results(i).Labels = labels; 
    end
end