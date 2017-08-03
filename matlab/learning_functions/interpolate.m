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