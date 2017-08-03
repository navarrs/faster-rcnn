%% Plot Layer Weights
function plotLayerWeights(weights, size, tit, filename)
   w = mat2gray(weights);
   w = imresize(w, [size size]); 
   a = figure;
   montage(w);
   title(tit); 
   saveas(a, filename, 'jpg');  
end 

