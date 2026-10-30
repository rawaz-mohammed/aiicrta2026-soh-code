%% REVIEWER2_POINT5_NORMALIZATION_ABLATION
% Reviewer 2's literal ask (#5 on their own priority list): "An ablation
% of the normalization, with capacity- and current-based features
% normalized too." Full comment: "The 'chemistry-agnostic' label also
% oversells the features. Only the voltage features are normalized.
% Charge delivered, duration and the current features are in absolute
% units, so they depend on capacity and C-rate... Because there's no
% ablation with and without normalization, the paper never shows that
% the normalization scheme, listed as its first contribution, actually
% helps."
%
% This script rebuilds the EXACT same NASA training table as the
% published paper (same nasa_feature_extraction.m logic, same filters,
% ORIGINAL BOL-relative SOH definition -- for direct comparability with
% the published Table II, isolating normalization as the only changed
% variable), then additionally computes each battery's own BOL capacity
% so that current- and charge-based features can be expressed as
% capacity-normalized ratios (the standard "C-rate" convention:
% current / nameplate-or-BOL capacity), instead of their raw absolute
% units. Duration_h is left unchanged (no clean capacity-based
% normalization exists for a timing feature without extra assumptions --
% documented explicitly rather than forcing an arbitrary one).
%
% Trains GPR + LSBoost on the FULLY-normalized feature set (5-fold CV +
% external validation on the real hardware, same methodology as
% train_soh_models.m) and reports results for direct comparison against
% the published numbers (partial normalization).

clear; clc; close all;
rng(42);

metadataFile = 'D:\NASA AI\archive\cleaned_dataset\metadata.csv';
dataFolder   = 'D:\NASA AI\archive\cleaned_dataset\data';
realDataFile = 'C:\Users\user\Desktop\Claude outputs\health_indicator_features_corrected.csv';
outDir       = 'C:\Users\user\Desktop\Claude outputs';

NASA_NOMINAL_VOLTAGE_V = 3.7;
MIN_BOL_CAPACITY_AH = 1.0;
MIN_SOH_PERCENT     = 30;
HARDWARE_NAMEPLATE_AH = 1.8;   % medium-life battery's own nameplate rating (per cell)

%% ---- Phase 1: rebuild the exact training table (same logic as nasa_feature_extraction.m), plus BOL_Capacity_Ah per row ----
fprintf('Phase 1: rebuilding NASA training table from raw cycles (+ BOL capacity per row)...\n');

meta = readtable(metadataFile);
meta.battery_id = string(meta.battery_id);
meta.type       = string(meta.type);
meta.filename   = string(meta.filename);
meta.Capacity   = str2double(string(meta.Capacity));

batteries = unique(meta.battery_id);
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
        continue
    end

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
        pEarlyV = polyfit(t(1:k), V(1:k), 1);
        pEarlyI = polyfit(t(1:k), I(1:k), 1);
        pLateI  = polyfit(t(end-k+1:end), I(end-k+1:end), 1);
        Ah = trapz(t, I) / 3600;

        startV_ratio          = V(1)   / NASA_NOMINAL_VOLTAGE_V;
        endV_ratio            = V(end) / NASA_NOMINAL_VOLTAGE_V;
        earlyVSlope_ratioPerH = (pEarlyV(1)*3600) / NASA_NOMINAL_VOLTAGE_V;

        row = table(battId, ambT, soh, duration_h, dTemp, max(Tp), ...
                    V(1), V(end), I(1), I(end), mean(I), ...
                    pEarlyV(1)*3600, pEarlyI(1)*3600, pLateI(1)*3600, Ah, ...
                    startV_ratio, endV_ratio, earlyVSlope_ratioPerH, bolCapacity, ...
                    'VariableNames', {'BatteryLabel','AmbientTemp_C','SOH_percent', ...
                                       'Duration_h','DeltaTemp_C','MaxTemp_C', ...
                                       'StartVoltage_V','EndVoltage_V', ...
                                       'StartCurrent_A','EndCurrent_A','MeanCurrent_A', ...
                                       'EarlyVoltageSlope_VpH','EarlyCurrentSlope_ApH', ...
                                       'LateCurrentSlope_ApH','ChargeDelivered_Ah', ...
                                       'StartVoltage_ratio','EndVoltage_ratio', ...
                                       'EarlyVoltageSlope_ratioPerH','BOL_Capacity_Ah'});
        allHI = [allHI; row]; %#ok<AGROW>
    end
end

fprintf('Rebuilt training table: %d rows, %d batteries\n', height(allHI), numel(unique(allHI.BatteryLabel)));

%% ---- Phase 2: capacity-normalize the current- and charge-based features ----
allHI.CRate_Start            = allHI.StartCurrent_A          ./ allHI.BOL_Capacity_Ah;
allHI.CRate_End               = allHI.EndCurrent_A            ./ allHI.BOL_Capacity_Ah;
allHI.CRate_Mean               = allHI.MeanCurrent_A           ./ allHI.BOL_Capacity_Ah;
allHI.EarlyCurrentSlope_ratio = allHI.EarlyCurrentSlope_ApH   ./ allHI.BOL_Capacity_Ah;
allHI.LateCurrentSlope_ratio  = allHI.LateCurrentSlope_ApH    ./ allHI.BOL_Capacity_Ah;
allHI.ChargeDelivered_ratio   = allHI.ChargeDelivered_Ah      ./ allHI.BOL_Capacity_Ah;

writetable(allHI, fullfile(outDir, 'nasa_health_indicator_features_ablation.csv'));
fprintf('Saved: nasa_health_indicator_features_ablation.csv\n');

%% ---- Phase 3: 5-fold CV with the FULLY-normalized predictor set ----
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

fprintf('\nAblation training set: %d rows, %d predictors (fully normalized)\n', size(X,1), size(X,2));

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

fprintf('\n=== Ablation 5-fold CV (fully-normalized features) ===\n');
fprintf('GPR    : RMSE = %.2f%% (+-%.2f), MAE = %.2f%%\n', mean(rmseGPR), std(rmseGPR), mean(maeGPR));
fprintf('LSBoost: RMSE = %.2f%% (+-%.2f), MAE = %.2f%%\n', mean(rmseLSB), std(rmseLSB), mean(maeLSB));
fprintf('\n(Published, partial normalization only: GPR RMSE 2.54%% (+-0.51), MAE 0.94%%; LSBoost RMSE 4.08%% (+-0.36), MAE 2.58%%)\n');

%% ---- Phase 4: train FINAL models on full ablation set, external validation on real hardware ----
mdlGPR_final = fitrgp(X, y, 'KernelFunction', 'ardsquaredexponential', 'Standardize', true);
mdlLSB_final = fitrensemble(X, y, 'Method', 'LSBoost', ...
    'NumLearningCycles', 100, 'Learners', templateTree('MaxNumSplits', 10));

Treal = readtable(realDataFile);
Treal.CRate_Start             = Treal.StartCurrent_A        ./ HARDWARE_NAMEPLATE_AH;
Treal.CRate_End                = Treal.EndCurrent_A          ./ HARDWARE_NAMEPLATE_AH;
Treal.CRate_Mean                = Treal.MeanCurrent_A         ./ HARDWARE_NAMEPLATE_AH;
Treal.EarlyCurrentSlope_ratio  = Treal.EarlyCurrentSlope_ApH ./ HARDWARE_NAMEPLATE_AH;
Treal.LateCurrentSlope_ratio   = Treal.LateCurrentSlope_ApH  ./ HARDWARE_NAMEPLATE_AH;
Treal.ChargeDelivered_ratio    = Treal.ChargeDelivered_Ah    ./ HARDWARE_NAMEPLATE_AH;

Xreal = Treal{:, predictorNamesFull};
trueSOH = Treal.SOH_percent;

predGPR_real = predict(mdlGPR_final, Xreal);
predLSB_real = predict(mdlLSB_final, Xreal);

resultTable = table(Treal.BatteryLabel, Treal.AmbientTemp_C, trueSOH, ...
    predGPR_real, abs(predGPR_real - trueSOH), ...
    predLSB_real, abs(predLSB_real - trueSOH), ...
    'VariableNames', {'BatteryLabel','AmbientTemp_C','True_SOH_percent', ...
                       'GPR_Predicted_SOH','GPR_AbsError', ...
                       'LSBoost_Predicted_SOH','LSBoost_AbsError'});

fprintf('\n=== Ablation external validation on real hardware (fully-normalized features) ===\n');
disp(resultTable)
fprintf('(Published, partial normalization: GPR error 11.8pp both rows; LSBoost error 18.6pp @28C, 20.4pp @18C)\n');

writetable(resultTable, fullfile(outDir, 'reviewer2_point5_ablation_external_results.csv'));
fprintf('\nSaved: reviewer2_point5_ablation_external_results.csv\n');
fprintf('\n=== DONE ===\n');
