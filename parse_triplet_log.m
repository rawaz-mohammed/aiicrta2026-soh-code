function T = parse_triplet_log(currentFile, voltageFile, tempFile)
%PARSE_TRIPLET_LOG Read a charge/discharge run logged as three separate
%two-column text files (time(s), value), as exported from the Arduino
%rig / COMSOL-import files, e.g.:
%   current_vs_time.txt        : % time(s)  current(A)
%   voltage_vs_time_1.txt      : % time(s)  voltage(V)
%   temperature_vs_time_2.txt  : % time(s)  temperature(C)
%
%   T = PARSE_TRIPLET_LOG(currentFile, voltageFile, tempFile) returns a
%   timetable T with variables Current_A, Voltage_V, Temp_C, sampled on
%   the current file's time grid (voltage/temperature are linearly
%   interpolated onto it if their timestamps differ slightly).
%
%   Example:
%       T = parse_triplet_log('current_vs_time.txt', ...
%                              'voltage_vs_time_1.txt', ...
%                              'temperature_vs_time_2.txt');

    [tI, I]  = readTwoColumn(currentFile);
    [tV, V]  = readTwoColumn(voltageFile);
    [tT, Tp] = readTwoColumn(tempFile);

    tMin = max([tI(1), tV(1), tT(1)]);
    tMax = min([tI(end), tV(end), tT(end)]);
    keep = tI >= tMin & tI <= tMax;

    t  = tI(keep);
    Ic = I(keep);
    Vc = interp1(tV, V, t, 'linear');
    Tc = interp1(tT, Tp, t, 'linear');

    Time = seconds(t);
    T = timetable(Time, Ic, Vc, Tc, ...
                  'VariableNames', {'Current_A','Voltage_V','Temp_C'});

    fprintf('Parsed %d aligned samples, %.0f s (%.2f h)\n', ...
            numel(t), t(end)-t(1), (t(end)-t(1))/3600);
end

function [t, v] = readTwoColumn(filename)
    if ~isfile(filename)
        error('parse_triplet_log:fileNotFound', 'File not found: %s', filename);
    end
    fid = fopen(filename, 'r');
    raw = textscan(fid, '%f %f', 'CommentStyle', '%', 'CollectOutput', true);
    fclose(fid);
    data = raw{1};
    t = data(:,1);
    v = data(:,2);
end
