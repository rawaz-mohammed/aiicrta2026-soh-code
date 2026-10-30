%% REVIEWER2_POINT5_CV_ONLY
% Re-derives just the 5-fold CV numbers from the already-saved ablation
% feature table (nasa_health_indicator_features_ablation.csv), writing
% them to a CSV instead of relying on console output (screen capture was
% unreliable this run). Uses the exact same rng(42)/cvpartition/model
% settings as the main ablation script.

clear; clc;
rng(42);

outDir = 'C:\Users\user\Desktop\Claude outputs';
allHI = readtable(fullfile(outDir, 'nasa_health_indicator_features_ablation.csv'));

predictorNamesFull = {'AmbientTemp_C','Duration_h','DeltaTemp_C','MaxTemp_C', ...
                       'StartVoltage_ratio','EndVoltage_ratio', ...
                       'CRate_Start','CRate_End','CRate_Mean', ...
                       'EarlyVoltageSlope_ratioPerH','EarlyCurrentSlope_ratio', ...
                       'LateCurrentSlope_ratio','ChargeDelivered_ratio'};

X = allHI{:, predictorNamesFull};
y = allHI.SOH_percent;
goodRows = all(isfinite(X), 2) & isfinite(y);
X = X(goodRows, :);
y = y(goodRows);

K = 5;
cv = cvpartition(size(X,1), 'KFold', K);

rmseGPR = zeros(K,1); maeGPR = zeros(K,1);
rmseLSB = zeros(K,1); maeLSB = zeros(K,1);

for i = 1:K
    trIdx = training(cv,i); teIdx = test(cv,i);
    mdlGPR_i = fitrgp(X(trIdx,:), y(trIdx), 'KernelFunction', 'ardsquaredexponential', 'Standardize', true);
    predGPR = predict(mdlGPR_i, X(teIdx,:));
    rmseGPR(i) = sqrt(mean((predGPR - y(teIdx)).^2));
    maeGPR(i)  = mean(abs(predGPR - y(teIdx)));

    mdlLSB_i = fitrensemble(X(trIdx,:), y(trIdx), 'Method', 'LSBoost', 'NumLearningCycles', 100, 'Learners', templateTree('MaxNumSplits', 10));
    predLSB = predict(mdlLSB_i, X(teIdx,:));
    rmseLSB(i) = sqrt(mean((predLSB - y(teIdx)).^2));
    maeLSB(i)  = mean(abs(predLSB - y(teIdx)));
end

summaryTable = table({'GPR';'LSBoost'}, [mean(rmseGPR); mean(rmseLSB)], [std(rmseGPR); std(rmseLSB)], ...
    [mean(maeGPR); mean(maeLSB)], 'VariableNames', {'Model','MeanRMSE','StdRMSE','MeanMAE'});
writetable(summaryTable, fullfile(outDir, 'reviewer2_point5_cv_summary.csv'));
disp(summaryTable)
fprintf('SAVED_OK\n');
