%% Interpolation of activation map 
function map = interpolate(image_map) 
 % Interpolate values
    map = image_map; 
    for i = 1:size(image_map, 1)
         for j = 1:size(image_map, 2)
             if image_map(i, j) < 10
                map(i, j) = 0;
             else
                map(i, j) = image_map(i, j);
             end
         end
    end
end

%% Evaluate predictions 
function results = eval_pred(data)
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

%% Plot precision - recall curve 
function prec_recall(precision, recall, av, name, path)
    % PRECISION: relevant instances among retrieved instances
    % RECALL: relevant instances that have been retrieved over total 
    %         relevant instances in the image
    figure    
    a = plot(recall, precision);
    xlabel('Recall')
    ylabel('Precision')
    grid on
    title(sprintf(strcat(name, 'average precision = %.3f'), av(1))); 
    saveas(a, fullfile(path, strcat('precision_recall_', name)), 'jpg');
end

%% Get F1 score and plot it
% Get F1 score
function f1_score(precision, recall, name, path) 
    f1 = zeros(int16(length(recall)), 1);  
    for i = 1:int16(length(precision))
         f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    end 
    figure
    a = plot(f1);
    xlabel('Iteration')
    ylabel('F1-score')
    axis([1 int16(length(recall)) 0 1]);
    grid on
    title(sprintf(strcat(name, '_f1_score'))); 
    saveas(a, fullfile(path, strcat(name, '_f1 score')), 'jpg');
end

%% Plot Layer Weights
function plotLayerWeights(weights, size, title, filename)
   w = mat2gray(weights);
   w = imresize(w, [size size]); 
   a = figure;
   montage(w);
   title(title); 
   saveas(a, filename, 'jpg');  
end 

%% Subplots
function plot_save(i1, t1, i2, t2, i3, t3, i4, t4, i5, t5, i6, t6, in, n)
        % Plot figure 
        fig = figure;
        subplot(2,3,1), x = imresize(i1, 2); imshow(x); title(t1, 'FontSize', 9);
        subplot(2,3,4), x = imresize(i2, 2); imshow(x); title(t2, 'FontSize', 9); 
        subplot(2,3,2), imshow(i3); title(t3, 'FontSize', 9);
        subplot(2,3,5), imshow(i4); title(i4, 'FontSize', 9);
        subplot(2,3,3), imshow(i5); title(i5,'FontSize', 9 ); 
        subplot(2,3,6), imshow(i6); title(i6, 'FontSize', 9);
    
        % Save figure
        image_name = strcat(in, num2str(n), '.jpg');
        test_image = strcat(pwd, '/', test_folder, '/test_images/');
        saveas(fig, fullfile(test_image, image_name), 'jpg');
    end  