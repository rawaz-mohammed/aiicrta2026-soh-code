%% REVIEWER2_POINT3_LOBO_CV
% Reviewer 2's literal ask (#3 on their own priority list): "Cross-
% validation that holds out whole batteries." Full comment: "The
% cross-validation probably flatters both models, GPR in particular.
% Splitting cycles at random puts neighbouring, nearly identical cycles
% from the same battery into both training and test folds. A smooth
% interpolator like GPR does very well under those conditions, which
% likely explains its MAE of 0.94%. The paper's whole argument is about
% predicting an unseen battery, so the CV should hold out whole batteries
% (leave-one-battery-out or similar)."
%
% This script rebuilds the EXACT same NASA training table as the
% published paper (same nasa_feature_extraction.m logic, same filters,
% same BOL-relative SOH definition -- NOT the Point-2 corrected
% definition, so results are directly comparable to the published
% Table II), then replaces the published random 5-fold CV with
% leave-one-battery-out (LOBO) CV: for each battery, train on every OTHER
% battery's cycles and test only on that battery's own held-out cycles.

clear; clc; close all;
rng(42);

metadataFile = 'D:\NASA AI\archive\cleaned_dataset\metadata.csv';
dataFolder   = 'D:\NASA AI\archive\cleaned_dataset\data';
outDir       = 'C:\Users\user\Desktop\Claude outputs';

NASA_NOMINAL_VOLTAGE_V = 3.7;
MIN_BOL_CAPACITY_AH = 1.0;
MIN_SOH_PERCENT     = 30;

%% ---- Phase 1: rebuild the exact training table (same logic as nasa_feature_extraction.m) ----
fprintf('Phase 1: rebuilding NASA training table from raw cycles...\n');

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
    end
end

fprintf('Rebuilt training table: %d rows, %d batteries\n', height(allHI), numel(unique(allHI.BatteryLabel)));

%% ---- Phase 2: leave-one-battery-out CV ----
predictorNames = {'AmbientTemp_C','Duration_h','DeltaTemp_C','MaxTemp_C', ...
                   'StartVoltage_ratio','EndVoltage_ratio','StartCurrent_A','EndCurrent_A', ...
                   'MeanCurrent_A','EarlyVoltageSlope_ratioPerH','EarlyCurrentSlope_ApH', ...
                   'LateCurrentSlope_ApH','ChargeDelivered_Ah'};

X = allHI{:, predictorNames};
y = allHI.SOH_percent;
battLabels = allHI.BatteryLabel;
goodRows = all(isfinite(X),2) & isfinite(y);
X = X(goodRows,:); y = y(goodRows); battLabels = battLabels(goodRows);

uniqueBatt = unique(battLabels);
nB = numel(uniqueBatt);
fprintf('\nPhase 2: running leave-one-battery-out CV across %d batteries...\n', nB);

rmseGPR_byBatt = nan(nB,1); maeGPR_byBatt = nan(nB,1); nTestByBatt = nan(nB,1);
rmseLSB_byBatt = nan(nB,1); maeLSB_byBatt = nan(nB,1);
allResidGPR = []; allResidLSB = [];

for i = 1:nB
    testMask = battLabels == uniqueBatt(i);
    trainMask = ~testMask;
    if sum(testMask) == 0
        continue
    end

    mdlGPR = fitrgp(X(trainMask,:), y(trainMask), ...
        'KernelFunction','ardsquaredexponential','Standardize',true);
    predGPR = predict(mdlGPR, X(testMask,:));
    residGPR = predGPR - y(testMask);
    rmseGPR_byBatt(i) = sqrt(mean(residGPR.^2));
    maeGPR_byBatt(i)  = mean(abs(residGPR));
    allResidGPR = [allResidGPR; residGPR]; %#ok<AGROW>

    mdlLSB = fitrensemble(X(trainMask,:), y(trainMask), 'Method','LSBoost', ...
        'NumLearningCycles',100,'Learners',templateTree('MaxNumSplits',10));
    predLSB = predict(mdlLSB, X(testMask,:));
    residLSB = predLSB - y(testMask);
    rmseLSB_byBatt(i) = sqrt(mean(residLSB.^2));
    maeLSB_byBatt(i)  = mean(abs(residLSB));
    allResidLSB = [allResidLSB; residLSB]; %#ok<AGROW>

    nTestByBatt(i) = sum(testMask);
    fprintf('  [%d/%d] battery %s (n=%d test rows): GPR RMSE=%.2f%%, LSBoost RMSE=%.2f%%\n', ...
        i, nB, uniqueBatt(i), nTestByBatt(i), rmseGPR_byBatt(i), rmseLSB_byBatt(i));
end

fprintf('\n=== LOBO CV results (pooled across all %d out-of-fold predictions) ===\n', numel(allResidGPR));
fprintf('GPR     : RMSE = %.2f%%, MAE = %.2f%%\n', sqrt(mean(allResidGPR.^2)), mean(abs(allResidGPR)));
fprintf('LSBoost : RMSE = %.2f%%, MAE = %.2f%%\n', sqrt(mean(allResidLSB.^2)), mean(abs(allResidLSB)));

fprintf('\n=== LOBO CV results (mean +- std ACROSS BATTERIES, Table-II style) ===\n');
fprintf('GPR     : RMSE = %.2f%% (+-%.2f), MAE = %.2f%%\n', mean(rmseGPR_byBatt,'omitnan'), std(rmseGPR_byBatt,'omitnan'), mean(maeGPR_byBatt,'omitnan'));
fprintf('LSBoost : RMSE = %.2f%% (+-%.2f), MAE = %.2f%%\n', mean(rmseLSB_byBatt,'omitnan'), std(rmseLSB_byBatt,'omitnan'), mean(maeLSB_byBatt,'omitnan'));

fprintf('\n(Published, random cycle-level 5-fold CV, for comparison: GPR RMSE 2.54%% (+-0.51), MAE 0.94%%; LSBoost RMSE 4.08%% (+-0.36), MAE 2.58%%)\n');

resultByBatt = table(uniqueBatt, nTestByBatt, rmseGPR_byBatt, maeGPR_byBatt, rmseLSB_byBatt, maeLSB_byBatt, ...
    'VariableNames', {'BatteryLabel','N_test_rows','GPR_RMSE','GPR_MAE','LSBoost_RMSE','LSBoost_MAE'});
writetable(resultByBatt, fullfile(outDir, 'reviewer2_point3_lobo_results.csv'));
fprintf('\nSaved: %s\n', fullfile(outDir, 'reviewer2_point3_lobo_results.csv'));
fprintf('\n=== DONE ===\n');
