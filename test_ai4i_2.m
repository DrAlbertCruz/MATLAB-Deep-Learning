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
display( mean( [ results(1).time, results(1).time, results(1).time ] ) ./ 60 ./60 );

labelNames = unique( results(1).prediction );
for fold = 1:3
    for label = 1:length(labelNames)
        TP = sum( results(fold).prediction == labelNames(label) & ...
            results(fold).groundTruth == labelNames(label) );
        TN = sum( results(fold).prediction ~= labelNames(label) & ...
            results(fold).groundTruth ~= labelNames(label) );
        FP = sum( results(fold).prediction == labelNames(label) & ...
            results(fold).groundTruth ~= labelNames(label) );
        FN = sum( results(fold).prediction ~= labelNames(label) & ...
            results(fold).groundTruth == labelNames(label) );
        P = sum( results(fold).groundTruth == labelNames(label) );
        N = sum( results(fold).groundTruth ~= labelNames(label) );
        TPR = TP / P;
        TNR = TN / N;
        PPV = TP / ( TP + FP );
        NPV = TN / ( TN + FN );
        FNR = 1 - TPR;
        FPR = 1 - TPR;
        FDR = 1 - PPV;
        FOR = 1 - NPV;
        ACC = ( TP + TN ) / ( P + N );
        F1 = 2 * TP / (2 * TP + FP + FN );
        MCC = ( TP * TN - FP * FN ) / sqrt( ( TP + FP ) * (TP + FN ) * (TN + FP ) * (TN + FN) );
        metrics(label,1,fold) = TPR;
        metrics(label,2,fold) = TNR;
        metrics(label,3,fold) = PPV;
        metrics(label,4,fold) = NPV;
        metrics(label,5,fold) = FNR;
        metrics(label,6,fold) = FPR;
        metrics(label,7,fold) = ACC;
        metrics(label,8,fold) = F1;
        metrics(label,9,fold) = MCC;
    end
end

metrics1 = mean( metrics, 3 );
stds = std( metrics, 0, 3 );