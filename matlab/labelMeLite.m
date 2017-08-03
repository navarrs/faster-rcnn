% read labeled image
%a = imread('mergedSet/extA_S_R_0_infer.png');
%a = imread('~/Video3.jpg');
a = imread('GabyWheelchair.png');
%a = imread('loneWheelchair.png');

% load label inference data array
%b = load('mergedSet/extA_S_R_0_infer.txt','ascii');
b = a;

close all

% display labeled image, to define polygons
figure(2), imagesc( a ),  hold on

% Get ready to define polygons
while(1)
    
    % Enter label number
    display('** Zoom in and align the image before entering the polygon **')
    labelIndex = input('Enter label index: ');
    count = 1;
    Xv = [];
    Yv = [];
    
    % capture the points defining the polygon
    while(1)
        [X, Y, mouseButton] = ginput(1)
        
        if( mouseButton == 3)
            break;
        end
        
        Xv( count ) = round( X );
        Yv( count ) = round( Y );
        
        count = count + 1;
        
        if(count > 2)
            plot( [Xv(count-1) Xv(count-2)], [Yv(count-1) Yv(count-2)], 'm',...
                'Linewidth', 3);
        end
        
    end
    
    
    % modify the inference data array coorespondingly
    %[ Xi, Yi ] = meshgrid( 1:320, 1:240 );
    [ Xi, Yi ] = meshgrid( size( a ) );
    
    Xi = Xi(:);
    Yi = Yi(:);
    
    IN = inpolygon( Xi(:), Yi(:), Xv, Yv );
    
    for k=1:length( Xi )
        if( IN(k) )
            b( Yi(k), Xi(k), : ) = labelIndex;
        end
    end
    
    % update the image with the new polygon
    
    figure( 1 ), clf,
    imagesc( b )
    
    doAnotherOne = input('Do you want to repeat the process? <1/0> ');
    
    if doAnotherOne == 0
        break;
    end
    figure(2), clf, imagesc( a ),  hold on
end