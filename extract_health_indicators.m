function HI = extract_health_indicators(T, ambientTemp_C, batteryLabel, soh_percent, pairCurrent, nominalVoltage_V)
%EXTRACT_HEALTH_INDICATORS Compute scalar health indicators (HI) from one
%parsed charge run (see PARSE_TRIPLET_LOG or PARSE_BATTERY_LOG) for
%feature-based / data-driven SOH estimation.
%
%   HI = EXTRACT_HEALTH_INDICATORS(T, ambientTemp_C, batteryLabel, ...
%                                   soh_percent, pairCurrent, nominalVoltage_V)
%
%   T                : timetable with Current_A, Voltage_V, Temp_C
%   ambientTemp_C    : room temperature for this run (e.g. 28 or 18)
%   batteryLabel     : e.g. "medium_life"
%   soh_percent      : ground-truth SOH (%) for this battery, e.g. 78.1
%   pairCurrent      : true if the logged current is the SUM over a
%                      2-cell parallel branch (2S2P pack) and should be
%                      halved to get per-cell current. Default true.
%   nominalVoltage_V : this cell's NOMINAL voltage (e.g. 3.2 for LiFePO4).
%                      Used to compute chemistry-agnostic voltage-RATIO
%                      features (see note below). Default 3.2 (LFP).
%
%   ---- WHY VoltageRatio FEATURES EXIST ----
%   Raw voltage-based indicators (Start/EndVoltage_V, voltage slope in
%   V/h) are NOT directly comparable across different cell chemistries:
%   e.g. "3.6 V" means "nearly full" for an LFP cell (nominal 3.2 V) but
%   "nearly empty" for a LiCoO2/NMC cell (nominal ~3.7 V, charged to
%   4.2 V) -- exactly the mismatch between this LFP hardware and the
%   NASA PCoE (LiCoO2/NMC-type) public training data. Dividing by the
%   cell's own nominal voltage puts both chemistries on a roughly
%   comparable scale, since charge cutoff / nominal voltage ratios are
%   similar (~1.1-1.15) across most Li-ion chemistries. The RAW Volt
%   columns are still returned too (useful for your own plots/tables),
%   but the *_ratio columns are what should be fed to the ML models so
%   the NASA-trained model generalizes to this LFP hardware.
%
%   Returns a 1-row table of indicators used in the literature for
%   partial-charging-curve / feature-based SOH estimation:
%       Duration_h                    total logged duration (hours)
%       DeltaTemp_C                    temperature change over the run
%       MaxTemp_C                      peak temperature reached
%       StartVoltage_V / EndVoltage_V  raw volts (reference/plots only)
%       StartCurrent_A / EndCurrent_A / MeanCurrent_A   (per cell)
%       EarlyVoltageSlope_VpH          dV/dt over the first 20% of samples
%       EarlyCurrentSlope_ApH          dI/dt over the first 20% of samples
%       LateCurrentSlope_ApH           dI/dt over the last 20% of samples
%                                      (CV-taper decay rate -- rises in
%                                       magnitude with internal-resistance /
%                                       aging changes)
%       ChargeDelivered_Ah             Coulomb-counted charge over the run
%                                      (per cell)
%       StartVoltage_ratio / EndVoltage_ratio / EarlyVoltageSlope_ratioPerH
%                                      chemistry-agnostic versions
%                                      (Volt-based value / nominalVoltage_V)
%                                      -- USE THESE as ML predictors
%
%   Example:
%       T  = parse_triplet_log('current_vs_time.txt', ...
%                               'voltage_vs_time_1.txt', ...
%                               'temperature_vs_time_2.txt');
%       HI = extract_health_indicators(T, 28, "medium_life", 78.1, true, 3.2);

    if nargin < 5
        pairCurrent = true;
    end
    if nargin < 6
        nominalVoltage_V = 3.2;   % LiFePO4 nominal cell voltage
    end

    t = seconds(T.Time - T.Time(1));

    I  = T.Current_A;
    V  = T.Voltage_V;
    Tp = T.Temp_C;

    if pairCurrent
        I = I / 2;
    end

    n = numel(t);
    k = max(5, round(0.2*n));

    duration_h = (t(end) - t(1)) / 3600;
    dTemp      = Tp(end) - Tp(1);

    pEarlyV = polyfit(t(1:k),     V(1:k),     1);
    pEarlyI = polyfit(t(1:k),     I(1:k),     1);
    pLateI  = polyfit(t(end-k+1:end), I(end-k+1:end), 1);

    Ah = trapz(t, I) / 3600;

    startV_ratio          = V(1)   / nominalVoltage_V;
    endV_ratio             = V(end) / nominalVoltage_V;
    earlyVSlope_ratioPerH  = (pEarlyV(1)*3600) / nominalVoltage_V;

    HI = table(string(batteryLabel), ambientTemp_C, soh_percent, duration_h, ...
               dTemp, max(Tp), V(1), V(end), I(1), I(end), mean(I), ...
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
end
