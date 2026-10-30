%% TRAIN_SOH_MODELS
% Trains the two agreed SOH-estimation models -- Gaussian Process
% Regression (GPR) as the proposed model, and a Boosted Trees Ensemble
% (LSBoost) as the comparison baseline -- on the NASA-derived training
% feature table, evaluates both with 5-fold cross-validation, then runs
% the paper's core result: external validation against YOUR real
% hardware measurements (medium-life battery, 28C and 18C charge runs,
% ground-truth SOH = 78.1% for both, from Coulomb counting).
%
% Requires: Statistics and Machine Learning Toolbox (fitrgp, fitrensemble)
%
% ---- BEFORE RUNNING, make sure BOTH files are in the same folder as
%      this script (or edit the two paths below) ----
%   1) nasa_health_indicator_features.csv   <- from nasa_feature_extraction.m
%   2) health_indicator_features.csv        <- your real 2-row validation
%                                               set, from main_pipeline_example.m

clear; clc; close all;
rng(42);   % fixed seed -> reproducible results (report this in the paper)

trainFile    = 'nasa_health_indicator_features.csv';
realDataFile = 'health_indicator_features.csv';

if ~isfile(trainFile)
    error('Training file not found: %s\nRun nasa_feature_extraction.m first.', trainFile);
end
if ~isfile(realDataFile)
    error('Real validation file not found: %s\nMake sure it is in the current folder.', realDataFile);
end

% NOTE: uses the chemistry-agnostic VoltageRatio features (Volt / nominal
% voltage), NOT the raw *_V columns -- your real hardware is LFP (nominal
% 3.2V) while the NASA training cells are LiCoO2/NMC-type (nominal
% ~3.7V, charge to ~4.2V). Raw volts are not comparable across that
% chemistry gap; the ratio versions are, and both CSVs (real + NASA)
% were built with matching *_ratio columns for exactly this reason.
predictorNames = {'AmbientTemp_C','Duration_h','DeltaTemp_C','MaxTemp_C', ...
                   'StartVoltage_ratio','EndVoltage_ratio','StartCurrent_A','EndCurrent_A', ...
                   'MeanCurrent_A','EarlyVoltageSlope_ratioPerH','EarlyCurrentSlope_ApH', ...
                   'LateCurrentSlope_ApH','ChargeDelivered_Ah'};

%% ---- 1) load training data (NASA) ----
Ttrain = readtable(trainFile);
X = Ttrain{:, predictorNames};
y = Ttrain.SOH_percent;

goodRows = all(isfinite(X), 2) & isfinite(y);   % drop any corrupt/incomplete cycles
X = X(goodRows, :);
y = y(goodRows);

fprintf('Training set: %d rows, %d predictors, SOH range %.1f%%-%.1f%%\n', ...
        size(X,1), size(X,2), min(y), max(y));

%% ---- 2) 5-fold cross-validation (reported in the paper as model performance) ----
K = 5;
cv = cvpartition(size(X,1), 'KFold', K);

rmseGPR = zeros(K,1); maeGPR = zeros(K,1);
rmseLSB = zeros(K,1); maeLSB = zeros(K,1);

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
end

fprintf('\n--- 5-fold cross-validation on NASA training data ---\n');
fprintf('GPR    : RMSE = %.2f%% (+-%.2f), MAE = %.2f%%\n', mean(rmseGPR), std(rmseGPR), mean(maeGPR));
fprintf('LSBoost: RMSE = %.2f%% (+-%.2f), MAE = %.2f%%\n', mean(rmseLSB), std(rmseLSB), mean(maeLSB));

%% ---- 3) train FINAL models on the full NASA training set ----
mdlGPR_final = fitrgp(X, y, 'KernelFunction', 'ardsquaredexponential', 'Standardize', true);
mdlLSB_final = fitrensemble(X, y, 'Method', 'LSBoost', ...
    'NumLearningCycles', 100, 'Learners', templateTree('MaxNumSplits', 10));

%% ---- 4) EXTERNAL VALIDATION: your real hardware data (the paper's core result) ----
Treal   = readtable(realDataFile);
Xreal   = Treal{:, predictorNames};
trueSOH = Treal.SOH_percent;

predGPR_real = predict(mdlGPR_final, Xreal);
predLSB_real = predict(mdlLSB_final, Xreal);

resultTable = table(Treal.BatteryLabel, Treal.AmbientTemp_C, trueSOH, ...
    predGPR_real, abs(predGPR_real - trueSOH), ...
    predLSB_real, abs(predLSB_real - trueSOH), ...
    'VariableNames', {'BatteryLabel','AmbientTemp_C','True_SOH_percent', ...
                       'GPR_Predicted_SOH','GPR_AbsError', ...
                       'LSBoost_Predicted_SOH','LSBoost_AbsError'});

fprintf('\n--- External validation on your REAL hardware data ---\n');
disp(resultTable)

writetable(resultTable, 'soh_validation_results.csv');
fprintf('\nSaved: soh_validation_results.csv\n');

%% ---- 5) comparison figure (grayscale bars, IEEE style, consistent with your other figures) ----
fig = figure('Color','w','Units','centimeters','Position',[2 2 14 10]);
b = bar([trueSOH, predGPR_real, predLSB_real]);
b(1).FaceColor = [0.15 0.15 0.15];
b(2).FaceColor = [0.55 0.55 0.55];
b(3).FaceColor = [0.85 0.85 0.85];

xt = 1:height(Treal);
labels = arrayfun(@(t) sprintf('%d^\\circC', t), Treal.AmbientTemp_C, 'UniformOutput', false);
set(gca, 'XTick', xt, 'XTickLabel', labels, ...
    'FontName', 'Times New Roman', 'FontSize', 11, 'Box', 'on');
ylabel('SOH (%)', 'FontName', 'Times New Roman', 'FontSize', 11);
legend({'True (Coulomb counting)', 'GPR prediction', 'LSBoost prediction'}, ...
       'Location', 'southoutside', 'FontName', 'Times New Roman', 'FontSize', 10);
ylim([0 100]);
grid on;

exportgraphics(fig, 'soh_validation_comparison.png', 'Resolution', 300);
fprintf('Saved: soh_validation_comparison.png\n');

% ---- Notes for the paper ----
% - Report the 5-fold CV RMSE/MAE as the models' general performance on
%   the NASA training distribution.
% - Report resultTable (True vs GPR vs LSBoost, both temperatures) as the
%   headline result: does the model trained on public data generalize to
%   your real second-life hardware, INCLUDING under a temperature (18C)
%   it saw only implicitly through the AmbientTemp_C feature?
% - rng(42) is fixed so these numbers are reproducible if you re-run.
