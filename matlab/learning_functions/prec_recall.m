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
    title(sprintf(strcat(name, ' average precision = %.3f'), av(1))); 
    saveas(a, fullfile(path, strcat('precision_recall_', name)), 'jpg');
end