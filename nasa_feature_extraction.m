%% NASA_FEATURE_EXTRACTION  (v2 -- matches the "cleaned_dataset" CSV layout)
% Builds the TRAINING feature table for the SOH regression model from the
% full NASA PCoE Li-ion Battery Aging Dataset, as you downloaded it:
%
%   C:\Users\user\Desktop\cleaned_dataset\metadata.csv   <- index: one row
%                                                            per cycle (type,
%                                                            battery_id,
%                                                            filename, Capacity...)
%   C:\Users\user\Desktop\cleaned_dataset\data\NNNNN.csv <- per-cycle time
%                                                            series (Voltage_measured,
%                                                            Current_measured,
%                                                            Temperature_measured, Time)
%
% This version covers ALL battery IDs found in metadata.csv automatically
% (dozens of cells across many ambient-temperature/loading conditions) --
% much richer than the original 4-battery plan, and needs no per-battery
% file list.
%
% ---- IMPORTANT METHODOLOGY NOTE (mention this in your Methodology section) ----
% For each CHARGE cycle, SOH_percent here = (capacity of the NEXT discharge
% cycle for that battery) / (capacity of that battery's FIRST recorded
% discharge cycle, i.e. its beginning-of-life/BOL capacity) * 100.
% This "capacity fade relative to BOL" definition is used because the many
% battery groups in this dataset do not share one nameplate/nominal
% rating. It is standard in the aging literature, but it is NOT identical
% to your real hardware's SOH definition (capacity / nameplate rating).
% State both definitions explicitly when you write up the paper.
%
% ---- CHEMISTRY NOTE ----
% These NASA cells are 18650 Li-ion cells charged up to ~4.2V (confirmed
% from your own 00001.csv: starts at 4.2468V) -- i.e. LiCoO2/NMC-type
% chemistry, NOT LiFePO4 like your real hardware (nominal 3.2V). Raw
% Volt-based features are therefore NOT directly comparable to your real
% data. This script computes chemistry-agnostic VoltageRatio features
% (Volt value / that chemistry's nominal voltage) alongside the raw Volt
% columns -- use the *_ratio columns as ML predictors, matching
% extract_health_indicators.m on the real-hardware side.
%
% ---- DATA-QUALITY FILTER (checked directly against your metadata.csv) ----
% Not all 34 battery IDs in this dataset are usable. Batteries B0005-B0032
% have realistic beginning-of-life (BOL) capacities (1.66-2.03 Ah) and
% smooth capacity fade -- these are kept. Batteries B0033 onward include
% several with implausible BOL capacities (some under 0.1 Ah -- not
% physically possible for an 18650 cell) and/or discharge cycles with
% Capacity = 0 mid-experiment (a failed measurement, not a real end-of-
% life reading) -- almost certainly different test protocols (partial
% cycling / impedance-focused tests), not simple full-capacity aging like
% your own hardware. Training on those unfiltered would teach the model
% fake near-0% SOH labels. MIN_BOL_CAPACITY_AH and MIN_SOH_PERCENT below
% filter these out; mention this filtering step in your Methodology.

clear; clc;

% You copied the dataset to your Desktop -- this path matches that.
metadataFile = 'C:\Users\user\Desktop\cleaned_dataset\metadata.csv';
dataFolder   = 'C:\Users\user\Desktop\cleaned_dataset\data';

NASA_NOMINAL_VOLTAGE_V = 3.7;   % standard consumer 18650 Li-ion nominal
                                 % voltage (LiCoO2/NMC-type) -- these NASA
                                 % cells charge to ~4.2V, consistent with
                                 % this chemistry, not LFP.

MIN_BOL_CAPACITY_AH = 1.0;   % skip batteries whose first-recorded
                             % capacity is below this (implausible for
                             % an 18650 cell -> different/corrupt protocol)
MIN_SOH_PERCENT     = 30;    % skip any charge cycle whose derived SOH
                             % label falls below this (almost certainly a
                             % failed Capacity=0 reading, not real aging)

if ~isfile(metadataFile)
    error('metadata.csv not found: %s\nEdit metadataFile above to match your path.', metadataFile);
end
if ~isfolder(dataFolder)
    error('data folder not found: %s\nEdit dataFolder above to match your path.', dataFolder);
end

meta = readtable(metadataFile);
meta.battery_id = string(meta.battery_id);
meta.type       = string(meta.type);
meta.filename   = string(meta.filename);

% IMPORTANT: 25 rows in metadata.csv have the literal text "[]" instead of
% a number in the Capacity column (a failed/empty capacity reading from
% the original source data) -- this makes MATLAB import the whole column
% as text instead of numbers. str2double() converts everything properly
% to double, turning "[]" and blanks into NaN (exactly what we want, so
% they're correctly skipped below), while leaving real numbers unchanged.
meta.Capacity = str2double(string(meta.Capacity));

batteries = unique(meta.battery_id);
fprintf('Found %d batteries in metadata.csv\n', numel(batteries));

allHI = table();

for bIdx = 1:numel(batteries)
    battId = batteries(bIdx);
    rows = meta(meta.battery_id == battId, :);
    rows = sortrows(rows, 'uid');   % chronological order within this battery

    dischargeCap = nan(height(rows), 1);
    for r = 1:height(rows)
        if rows.type(r) == "discharge" && ~isnan(rows.Capacity(r))
            dischargeCap(r) = rows.Capacity(r);
        end
    end
    firstDischargeIdx = find(~isnan(dischargeCap), 1, 'first');
    if isempty(firstDischargeIdx)
        continue   % no capacity data at all for this battery -- skip it
    end
    bolCapacity = dischargeCap(firstDischargeIdx);

    if bolCapacity < MIN_BOL_CAPACITY_AH
        fprintf('%s: skipped (BOL capacity %.3f Ah is implausible -- different test protocol)\n', battId, bolCapacity);
        continue
    end

    nRowsAdded = 0;
    for r = 1:height(rows)
        if rows.type(r) ~= "charge"
            continue
        end

        nextOffset = find(~isnan(dischargeCap(r+1:end)), 1, 'first');
        if isempty(nextOffset)
            continue   % no later discharge capacity to label this charge cycle with
        end
        soh = dischargeCap(r + nextOffset) / bolCapacity * 100;
        if soh < MIN_SOH_PERCENT || soh > 105
            continue   % implausible label (failed capacity reading), skip
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
            continue   % too short/corrupt a cycle to trust
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

writetable(allHI, 'nasa_health_indicator_features.csv');
fprintf('\nSaved nasa_health_indicator_features.csv (%d rows total, %d batteries)\n', ...
        height(allHI), numel(unique(allHI.BatteryLabel)));
fprintf('SOH range: %.1f%% to %.1f%%\n', min(allHI.SOH_percent), max(allHI.SOH_percent));

% ---- NOTE ON RUNTIME ----
% After the data-quality filter, this keeps ~12 batteries (B0005-B0032)
% and roughly 900-1000 charge cycles -- each read as its own CSV file
% (readtable has some overhead per file), so expect a few minutes to run,
% not the ~15-20 min a full unfiltered 34-battery run would have taken.
% That's normal, let it finish.

% ---- NEXT STEP ----
% Open train_soh_models.m and run it -- it now has a real, rich training
% set to work with, and will validate GPR and LSBoost against your real
% 78.1% SOH hardware measurements.
