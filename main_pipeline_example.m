%% MAIN_PIPELINE_EXAMPLE
% Builds the health-indicator feature table for the temperature-aware
% SOH study, from the medium-life battery's charge logs at 28C and 18C
% (each logged as three 2-column text files: current/voltage/temperature
% vs time). Ground-truth SOH = 78.1% (Coulomb-counted from the full
% discharge cycle at 28C, per-cell capacity 1.41 Ah vs 1.8 Ah nameplate).
%
% Point MATLAB's current folder at wherever you copied the two run
% folders ("middle battery chg 28 deg", "middle battery chg 18 deg"),
% or edit the paths below.

clear; clc;

SOH_GROUND_TRUTH = 78.1;   % from discharge-based Coulomb counting
NOMINAL_VOLTAGE_V = 3.2;   % LiFePO4 (LFP) nominal cell voltage -- used to
                           % build chemistry-agnostic voltage-ratio
                           % features (see extract_health_indicators.m),
                           % since the NASA training cells are a
                           % different chemistry (~3.7V nominal).

% Full absolute paths (Desktop) -- avoids "file not found" errors that
% happen when MATLAB's current folder isn't the folder these live in.
folder28 = 'C:\Users\user\Desktop\middle battery chg 28 deg';
folder18 = 'C:\Users\user\Desktop\middle battery chg 18 deg';

runs = {
    % currentFile, voltageFile, tempFile, ambientTemp_C
    fullfile(folder28,'current_vs_time.txt'), ...
    fullfile(folder28,'voltage_vs_time_1.txt'), ...
    fullfile(folder28,'temperature_vs_time_2.txt'), 28.0;

    fullfile(folder18,'Current_vs_Time_COMSOL_2.txt'), ...
    fullfile(folder18,'Voltage_vs_Time_COMSOL_2.txt'), ...
    fullfile(folder18,'Temperature_vs_Time_COMSOL_4.txt'), 18.0;
};

featureTable = table();

for k = 1:size(runs,1)
    curF = runs{k,1}; voltF = runs{k,2}; tempF = runs{k,3}; ambT = runs{k,4};

    if ~isfile(curF)
        warning('Skipping missing run (edit paths at top of script): %s', curF);
        continue
    end

    T  = parse_triplet_log(curF, voltF, tempF);
    HI = extract_health_indicators(T, ambT, "medium_life", SOH_GROUND_TRUTH, true, NOMINAL_VOLTAGE_V);
    featureTable = [featureTable; HI]; %#ok<AGROW>
end

disp(featureTable)

% --- plots for EACH run, auto-saved as publication-ready PNGs ---
% (These are the figures to drop straight into the paper -- one per
% ambient-temperature condition, so you can show the charge profile
% comparison at 28C vs 18C side by side.)
for k = 1:size(runs,1)
    curF = runs{k,1}; voltF = runs{k,2}; tempF = runs{k,3}; ambT = runs{k,4};
    if ~isfile(curF)
        continue
    end
    T = parse_triplet_log(curF, voltF, tempF);
    th = seconds(T.Time)/3600;

    fig = figure('Color','w','Position',[100 100 700 600]);
    subplot(3,1,1); plot(th, T.Voltage_V, 'LineWidth',1.3); ylabel('Voltage (V)'); grid on;
    title(sprintf('Medium-life battery charge @ %.0f^\\circC (pack/pair current)', ambT));
    subplot(3,1,2); plot(th, T.Current_A, 'LineWidth',1.3); ylabel('Current (A)'); grid on;
    subplot(3,1,3); plot(th, T.Temp_C, 'LineWidth',1.3);    ylabel('Temp (^\circC)'); grid on;
    xlabel('Time (h)');

    pngName = sprintf('charge_profile_%.0fC.png', ambT);
    exportgraphics(fig, pngName, 'Resolution', 300);  % 300 dpi, paper-ready
    fprintf('Saved figure: %s\n', pngName);
end

writetable(featureTable, 'health_indicator_features.csv');
fprintf('\nWrote health_indicator_features.csv (%d rows)\n', height(featureTable));

% --- next step ---
% This 2-row table (28C, 18C) is your EXTERNAL VALIDATION set.
% Train the actual SOH regression/classification model on a public
% dataset (NASA/CALCE/Oxford) using the same feature definitions, then
% check whether it predicts close to SOH_GROUND_TRUTH for both rows
% here despite their very different ChargeDelivered_Ah / temperature
% profiles -- that cross-temperature robustness check is the paper's
% core result.
