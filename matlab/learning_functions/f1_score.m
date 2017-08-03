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
    title(sprintf(strcat(name, ' f1 score'))); 
    saveas(a, fullfile(path, strcat(name, '_f1_score')), 'jpg');
end