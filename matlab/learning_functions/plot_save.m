%% Subplots
function plot_save(i1, t1, i2, t2, i3, t3, i4, t4, i5, t5, i6, t6, in, n, tf)
        % Plot figure 
        fig = figure;
        subplot(2,3,1), x = imresize(i1, 2); imshow(x); title(t1, 'FontSize', 9);
        subplot(2,3,4), x = imresize(i2, 2); imshow(x); title(t2, 'FontSize', 9); 
        subplot(2,3,2), imshow(i3); title(t3, 'FontSize', 9);
        subplot(2,3,5), imshow(i4); title(t4, 'FontSize', 9);
        subplot(2,3,3), imshow(i5); title(t5,'FontSize', 9 ); 
        subplot(2,3,6), imshow(i6); title(t6, 'FontSize', 9);
    
        % Save figure
        image_name = strcat(in, num2str(n), '.jpg');
        test_image = strcat(pwd, '/', tf, '/test_images/');
        saveas(fig, fullfile(test_image, image_name), 'jpg');
    end  