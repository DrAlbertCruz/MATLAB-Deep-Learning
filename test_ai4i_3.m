% Get results for figures
clear all
close all

FILE_TO_LOAD = 'ai4i_results_1.mat';
load( FILE_TO_LOAD );

results(1) = results1;
clear results1
results(2) = results2;
clear results2
results(3) = results3;
clear results3

display( 'Average run time' );
display( mean( [ results(1).time, results(1).time, results(1).time ] ) ./ 60 );
display( std( [ results(1).time, results(1).time, results(1).time ] ) ./ 60 );

labelNames = unique( results(1).prediction );
for x = 1:length(labelNames)
    for y = 1:length(labelNames)
        for fold = 1:3
            perf(fold) = sum( results(fold).prediction == labelNames(x) & ...
                results(fold).groundTruth == labelNames(y) ) ./ ...
                sum( results(fold).groundTruth == labelNames(y) );
        end
        confu( y, x ) = mean( perf );
        confs( y, x ) = std( perf );
    end
end