%% REVIEWER2_POINT2_SOH_DEFINITION
% Reviewer 2 asked to either (a) put the NASA training labels and the
% real-hardware label on the SAME SOH definition, or (b) estimate how
% large the offset between the two definitions is.
%
% The current NASA labels are BOL-relative: SOH = (next discharge
% capacity) / (that battery's own first-recorded/BOL discharge capacity).
% The real-hardware label is nameplate-relative: SOH = capacity /
% nameplate rating (1.8 Ah). NASA's own documented rated/nameplate
% capacity is 2.0 Ah (MathWorks' NASA-dataset documentation: "The rated
% capacity of the cells in this data set is 2 Ah").
%
% This script does option (a): it relabels every NASA training row onto
% the SAME nameplate-relative definition (SOH_nameplate = SOH_BOL *
% (that battery's own BOL capacity / 2.0 Ah)), computed already in Python
% from metadata.csv and written to
% nasa_health_indicator_features_nameplate.csv -- this script just loads
% that corrected file, retrains GPR/LSBoost exactly as before (same
% rng(42), same fitrgp/fitrensemble calls, same 13 predictors), and
% reports the 5-fold CV + external-validation numbers under the now
% CONSISTENT definition, alongside the original (BOL-relative,
% inconsistent) numbers for direct comparison.

clear; clc; close all;
outDir = 'C:\Users\user\Desktop\Claude outputs';

trainFile    = fullfile(outDir, 'nasa_health_indicator_features_nameplate.csv');  % CORRECTED labels
realDataFile = fullfile(outDir, 'health_indicator_features_corrected.csv');       % unchanged (already nameplate-relative)

predictorNames = {'AmbientTemp_C','Duration_h','DeltaTemp_C','MaxTemp_C', ...
                   'StartVoltage_ratio','EndVoltage_ratio','StartCurrent_A','EndCurrent_A', ...
                   'MeanCurrent_A','EarlyVoltageSlope_ratioPerH','EarlyCurrentSlope_ApH', ...
                   'LateCurrentSlope_ApH','ChargeDelivered_Ah'};

Ttrain = readtable(trainFile);
X = Ttrain{:, predictorNames};
y = Ttrain.SOH_percent;
goodRows = all(isfinite(X), 2) & isfinite(y);
X = X(goodRows, :);
y = y(goodRows);

fprintf('Training set (nameplate-relative labels): %d rows, %d predictors, SOH range %.1f%%-%.1f%%\n', ...
        size(X,1), size(X,2), min(y), max(y));

rng(42);
K = 5;
cv = cvpartition(size(X,1), 'KFold', K);

rmseGPR = zeros(K,1); maeGPR = zeros(K,1);
rmseLSB = zeros(K,1); maeLSB = zeros(K,1);
rmseTRIV = zeros(K,1); maeTRIV = zeros(K,1);

for i = 1:K
    trIdx = training(cv,i); teIdx = test(cv,i);

    mdlGPR_i = fitrgp(X(trIdx,:), y(trIdx), ...
        'KernelFunction', 'ardsquaredexponential', 'Standardize', true);
    predGPR = predict(mdlGPR_i, X(teIdx,:));
    rmseGPR(i) = sqrt(mean((predGPR - y(teIdx)).^2));
    maeGPR(i)  = mean(abs(predGPR - y(teIdx)));

    mdlLSB_i = fitrensemble(X(trIdx,:), y(trIdx), 'Method', 'LSBoost', ...
        'NumLearningCycles', 100, 'Learners', templateTree('MaxNumSplits', 10));
    predLSB = predict(mdlLSB_i, X(teIdx,:));
    rmseLSB(i) = sqrt(mean((predLSB - y(teIdx)).^2));
    maeLSB(i)  = mean(abs(predLSB - y(teIdx)));

    trivPred = repmat(mean(y(trIdx)), sum(teIdx), 1);
    rmseTRIV(i) = sqrt(mean((trivPred - y(teIdx)).^2));
    maeTRIV(i)  = mean(abs(trivPred - y(teIdx)));
end

fprintf('\n--- 5-fold CV under CORRECTED (nameplate-relative) NASA labels ---\n');
fprintf('GPR     : RMSE = %.2f%% (+-%.2f), MAE = %.2f%%\n', mean(rmseGPR), std(rmseGPR), mean(maeGPR));
fprintf('LSBoost : RMSE = %.2f%% (+-%.2f), MAE = %.2f%%\n', mean(rmseLSB), std(rmseLSB), mean(maeLSB));
fprintf('Trivial : RMSE = %.2f%% (+-%.2f), MAE = %.2f%%\n', mean(rmseTRIV), std(rmseTRIV), mean(maeTRIV));
fprintf('(Original, inconsistent-definition Table II for comparison: GPR RMSE 2.54%%, LSBoost RMSE 4.08%%)\n');

mdlGPR_final = fitrgp(X, y, 'KernelFunction', 'ardsquaredexponential', 'Standardize', true);
mdlLSB_final = fitrensemble(X, y, 'Method', 'LSBoost', ...
    'NumLearningCycles', 100, 'Learners', templateTree('MaxNumSplits', 10));
trivialMean_full = mean(y);

Treal   = readtable(realDataFile);
Xreal   = Treal{:, predictorNames};
trueSOH = Treal.SOH_percent;

[predGPR_real, sdGPR_real, yintGPR_real] = predict(mdlGPR_final, Xreal);
predLSB_real  = predict(mdlLSB_final, Xreal);
predTRIV_real = repmat(trivialMean_full, height(Treal), 1);

resultTable = table(Treal.BatteryLabel, Treal.AmbientTemp_C, trueSOH, ...
    predGPR_real, abs(predGPR_real - trueSOH), sdGPR_real, ...
    predLSB_real, abs(predLSB_real - trueSOH), ...
    predTRIV_real, abs(predTRIV_real - trueSOH), ...
    'VariableNames', {'BatteryLabel','AmbientTemp_C','True_SOH_percent', ...
                       'GPR_Predicted_SOH','GPR_AbsError','GPR_PredSD', ...
                       'LSBoost_Predicted_SOH','LSBoost_AbsError', ...
                       'Trivial_Predicted_SOH','Trivial_AbsError'});

fprintf('\n--- External validation under CORRECTED (nameplate-relative) labels ---\n');
disp(resultTable)

fprintf('\n=== BEFORE (original BOL-relative labels, published Table III) vs AFTER (nameplate-relative, consistent) ===\n');
fprintf('GPR     error: BEFORE 11.8 pp   ->  AFTER %.1f pp (both temperatures)\n', mean(abs(predGPR_real - trueSOH)));
fprintf('LSBoost error: BEFORE 18.6/20.4 pp (28C/18C) -> AFTER %.1f / %.1f pp\n', ...
    abs(predLSB_real(1)-trueSOH(1)), abs(predLSB_real(2)-trueSOH(2)));

writetable(resultTable, fullfile(outDir, 'reviewer2_point2_results.csv'));
fprintf('\nSaved: %s\n', fullfile(outDir, 'reviewer2_point2_results.csv'));
fprintf('\n=== DONE ===\n');
