FILE_NAME = 't_epochVal';
EPOCHS = 20;

result = zeros(2,EPOCHS);

for i=1:EPOCHS
    res = load( [ FILE_NAME num2str(i) '.mat' ] );
    this_result = zeros(1,5);
    for j=1:5
        this_result(j) = res.results(j).fold_results;
    end
    %% Remove outliers
    mm = mean( this_result );
    stdd = mean( this_result );
    
%     this_result( this_result > (mm + stdd) | ...
%         this_result < (mm - stdd) ) = [];
%     
%     this_result
    
    result(1,i) = mean( this_result );
    result(2,i) = var( this_result );
end

close all
figure(1);
plot( result(1,:), 'k-');
hold on, plot( result(1,:) + result(2,:), 'k--');
hold on, plot( result(1,:) - result(2,:), 'k--');