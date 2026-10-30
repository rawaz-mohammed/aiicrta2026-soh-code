%% REVIEWER2_POINT1_ANALYSIS
% Reproduces the exact NASA feature-extraction + GPR/LSBoost training
% pipeline used in the submitted paper (same logic as
% nasa_feature_extraction.m + train_soh_models.m, only the dataset path
% is corrected to its current location), then adds the two things
% Reviewer 2 asked for in comment 1:
%   (a) a trivial mean-prediction baseline, evaluated the same way as
%       GPR/LSBoost (5-fold CV on NASA + external validation on the real
%       hardware rows)
%   (b) GPR's predictive standard deviation / 95% interval on the two
%       real-hardware rows (to see whether GPR "knows" it is extrapolating)

clear; clc; close all;
outDir = 'C:\Users\user\Desktop\Claude outputs';

%% ---- Phase 1: NASA feature extraction (unchanged logic, corrected path) ----
metadataFile = 'D:\NASA AI\archive\cleaned_dataset\metadata.csv';
dataFolder   = 'D:\NASA AI\archive\cleaned_dataset\data';

NASA_NOMINAL_VOLTAGE_V = 3.7;
MIN_BOL_CAPACITY_AH = 1.0;
MIN_SOH_PERCENT     = 30;

if ~isfile(metadataFile)
    error('metadata.csv not found: %s', metadataFile);
end
if ~isfolder(dataFolder)
    error('data folder not found: %s', dataFolder);
end

meta = readtable(metadataFile);
meta.battery_id = string(meta.battery_id);
meta.type       = string(meta.type);
meta.filename   = string(meta.filename);
meta.Capacity = str2double(string(meta.Capacity));

batteries = unique(meta.battery_id);
fprintf('Found %d batteries in metadata.csv\n', numel(batteries));

allHI = table();

for bIdx = 1:numel(batteries)
    battId = batteries(bIdx);
    rows = meta(meta.battery_id == battId, :);
    rows = sortrows(rows, 'uid');

    dischargeCap = nan(height(rows), 1);
    for r = 1:height(rows)
        if rows.type(r) == "discharge" && ~isnan(rows.Capacity(r))
            dischargeCap(r) = rows.Capacity(r);
        end
    end
    firstDischargeIdx = find(~isnan(dischargeCap), 1, 'first');
    if isempty(firstDischargeIdx)
        continue
    end
    bolCapacity = dischargeCap(firstDischargeIdx);

    if bolCapacity < MIN_BOL_CAPACITY_AH
        fprintf('%s: skipped (BOL capacity %.3f Ah implausible)\n', battId, bolCapacity);
        continue
    end

    nRowsAdded = 0;
    for r = 1:height(rows)
        if rows.type(r) ~= "charge"
            continue
        end
        nextOffset = find(~isnan(dischargeCap(r+1:end)), 1, 'first');
        if isempty(nextOffset)
            continue
        end
        soh = dischargeCap(r + nextOffset) / bolCapacity * 100;
        if soh < MIN_SOH_PERCENT || soh > 105
            continue
        end

        cyclePath = fullfile(dataFolder, rows.filename(r));
        if ~isfile(cyclePath)
            continue
        end
        try
            d = readtable(cyclePath);
        catch
            continue
        end
        needCols = {'Voltage_measured','Current_measured','Temperature_measured','Time'};
        if ~all(ismember(needCols, d.Properties.VariableNames))
            continue
        end

        t = d.Time; V = d.Voltage_measured; I = d.Current_measured; Tp = d.Temperature_measured;
        valid = isfinite(t) & isfinite(V) & isfinite(I) & isfinite(Tp);
        t = t(valid); V = V(valid); I = I(valid); Tp = Tp(valid);
        [t, order] = sort(t); V = V(order); I = I(order); Tp = Tp(order);
        [t, uIdx] = unique(t, 'stable');
        V = V(uIdx); I = I(uIdx); Tp = Tp(uIdx);

        if numel(t) < 10
            continue
        end

        ambT = rows.ambient_temperature(r);
        n = numel(t);
        k = max(5, round(0.2*n));

        duration_h = (t(end) - t(1)) / 3600;
        dTemp      = Tp(end) - Tp(1);

        pEarlyV = polyfit(t(1:k),         V(1:k),         1);
        pEarlyI = polyfit(t(1:k),         I(1:k),         1);
        pLateI  = polyfit(t(end-k+1:end), I(end-k+1:end), 1);

        Ah = trapz(t, I) / 3600;

        startV_ratio          = V(1)   / NASA_NOMINAL_VOLTAGE_V;
        endV_ratio             = V(end) / NASA_NOMINAL_VOLTAGE_V;
        earlyVSlope_ratioPerH  = (pEarlyV(1)*3600) / NASA_NOMINAL_VOLTAGE_V;

        row = table(battId, ambT, soh, duration_h, dTemp, max(Tp), ...
                    V(1), V(end), I(1), I(end), mean(I), ...
                    pEarlyV(1)*3600, pEarlyI(1)*3600, pLateI(1)*3600, Ah, ...
                    startV_ratio, endV_ratio, earlyVSlope_ratioPerH, ...
                    'VariableNames', {'BatteryLabel','AmbientTemp_C','SOH_percent', ...
                                       'Duration_h','DeltaTemp_C','MaxTemp_C', ...
                                       'StartVoltage_V','EndVoltage_V', ...
                                       'StartCurrent_A','EndCurrent_A','MeanCurrent_A', ...
                                       'EarlyVoltageSlope_VpH','EarlyCurrentSlope_ApH', ...
                                       'LateCurrentSlope_ApH','ChargeDelivered_Ah', ...
                                       'StartVoltage_ratio','EndVoltage_ratio', ...
                                       'EarlyVoltageSlope_ratioPerH'});
        allHI = [allHI; row]; %#ok<AGROW>
        nRowsAdded = nRowsAdded + 1;
    end
    fprintf('%s: %d charge-cycle feature rows added\n', battId, nRowsAdded);
end

nasaCsvPath = fullfile(outDir, 'nasa_health_indicator_features.csv');
writetable(allHI, nasaCsvPath);
fprintf('\nSaved %s (%d rows total, %d batteries)\n', nasaCsvPath, height(allHI), numel(unique(allHI.BatteryLabel)));
fprintf('SOH range: %.1f%% to %.1f%%\n', min(allHI.SOH_percent), max(allHI.SOH_percent));

%% ---- Phase 2: training, 5-fold CV, trivial baseline (same predictors/seed as train_soh_models.m) ----
rng(42);

trainFile    = nasaCsvPath;
realDataFile = fullfile(outDir, 'health_indicator_features_corrected.csv');
% NOTE: the original health_indicator_features.csv on disk predates the
% voltage-ratio feature update (it has StartVoltage_V/StartCurrent_cell_A
% style columns, not the StartVoltage_ratio/StartCurrent_A columns
% predictorNames below expects -- a stale file from an earlier version of
% extract_health_indicators.m). health_indicator_features_corrected.csv
% is NOT a re-measurement: it is the same raw per-cell current/voltage
% values already in that file, with EarlyVoltageSlope_ratioPerH etc.
% deterministically recomputed as (raw value)/3.2 (LiFePO4 nominal cell
% voltage), i.e. exactly the formula in extract_health_indicators-1.m.

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

fprintf('\nTraining set: %d rows, %d predictors, SOH range %.1f%%-%.1f%%\n', ...
        size(X,1), size(X,2), min(y), max(y));

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

    % trivial baseline: predict the TRAINING FOLD's mean SOH for every test point
    trivPred = repmat(mean(y(trIdx)), sum(teIdx), 1);
    rmseTRIV(i) = sqrt(mean((trivPred - y(teIdx)).^2));
    maeTRIV(i)  = mean(abs(trivPred - y(teIdx)));
end

fprintf('\n--- 5-fold cross-validation on NASA training data ---\n');
fprintf('GPR     : RMSE = %.2f%% (+-%.2f), MAE = %.2f%%\n', mean(rmseGPR), std(rmseGPR), mean(maeGPR));
fprintf('LSBoost : RMSE = %.2f%% (+-%.2f), MAE = %.2f%%\n', mean(rmseLSB), std(rmseLSB), mean(maeLSB));
fprintf('Trivial : RMSE = %.2f%% (+-%.2f), MAE = %.2f%%  (predict training-fold mean SOH for every point)\n', mean(rmseTRIV), std(rmseTRIV), mean(maeTRIV));

%% ---- Phase 3: final models, external validation, trivial baseline, GPR predictive SD ----
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
    predGPR_real, abs(predGPR_real - trueSOH), sdGPR_real, yintGPR_real(:,1), yintGPR_real(:,2), ...
    predLSB_real, abs(predLSB_real - trueSOH), ...
    predTRIV_real, abs(predTRIV_real - trueSOH), ...
    'VariableNames', {'BatteryLabel','AmbientTemp_C','True_SOH_percent', ...
                       'GPR_Predicted_SOH','GPR_AbsError','GPR_PredSD','GPR_95CI_Lo','GPR_95CI_Hi', ...
                       'LSBoost_Predicted_SOH','LSBoost_AbsError', ...
                       'Trivial_Predicted_SOH','Trivial_AbsError'});

fprintf('\n--- External validation on REAL hardware data (incl. trivial baseline + GPR predictive SD) ---\n');
disp(resultTable)

fprintf('\nTrivial baseline = mean NASA training SOH = %.2f%% (predicted for both real-hardware rows, true = %.1f%%)\n', trivialMean_full, trueSOH(1));
fprintf('GPR 95%% prediction interval half-width on real-hardware rows: %.2f and %.2f percentage points\n', ...
    (yintGPR_real(1,2)-yintGPR_real(1,1))/2, (yintGPR_real(2,2)-yintGPR_real(2,1))/2);

writetable(resultTable, fullfile(outDir, 'reviewer2_point1_results.csv'));
fprintf('\nSaved: %s\n', fullfile(outDir, 'reviewer2_point1_results.csv'));
fprintf('\n=== DONE ===\n');
