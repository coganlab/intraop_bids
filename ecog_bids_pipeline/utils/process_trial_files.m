function process_trial_files(base_dir, subject_id, varargin)
% PROCESS_TRIAL_FILES Convert opaque data to strings in Trial_bids.mat files
%   This function finds all Trial_bids.mat files for a given subject, converts
%   any opaque data to strings, and saves the updated files.
%
%   Input:
%       base_dir    - Base directory containing subject folders
%       subject_id  - Subject ID (e.g., 'S41')
%   Optional Name-Value Pairs:
%       'FieldNames' - Cell array of field names to convert (default: all fields)
%       'Backup'     - Create backup before modifying (default: true)
%       'Verbose'    - Display progress (default: true)

% Parse input parameters
p = inputParser;
addRequired(p, 'base_dir', @ischar);
addRequired(p, 'subject_id', @ischar);
addParameter(p, 'FieldNames', {}, @iscellstr);
addParameter(p, 'Backup', true, @islogical);
addParameter(p, 'Verbose', true, @islogical);
parse(p, base_dir, subject_id, varargin{:});

% Find all Trial_bids.mat files for the subject
search_pattern = fullfile(base_dir, '**', subject_id, '**', 'Trial_bids.mat');
file_list = dir(search_pattern);

if isempty(file_list)
    fprintf('No Trial_bids.mat files found for subject %s in %s\n', subject_id, base_dir);
    return;
end

% Process each file
for i = 1:length(file_list)
    file_path = fullfile(file_list(i).folder, file_list(i).name);
    
    if p.Results.Verbose
        fprintf('Processing: %s\n', file_path);
    end
    
    % Load the data
    try
        data = load(file_path);
    catch ME
        warning('Failed to load %s: %s', file_path, ME.message);
        continue;
    end
    
    % Create backup if requested
    if p.Results.Backup
        backup_path = [file_path, '.bak'];
        if ~exist(backup_path, 'file')
            copyfile(file_path, backup_path);
            if p.Results.Verbose
                fprintf('  Created backup: %s\n', backup_path);
            end
        end
    end
    
    % Convert fields
    fields_to_convert = fieldnames(data);
    if ~isempty(p.Results.FieldNames)
        fields_to_convert = intersect(fields_to_convert, p.Results.FieldNames);
    end
    
    modified = false;
    for j = 1:length(fields_to_convert)
        field = fields_to_convert{j};
        
        if isstruct(data.(field)) && isfield(data.(field), 'trial')
            % Handle structure with trial field
            if p.Results.Verbose
                fprintf('  Converting field: %s.trial\n', field);
            end
            data.(field).trial = convert_trials(data.(field).trial, p.Results.Verbose);
            modified = true;
        else
            % Handle other fields
            if p.Results.Verbose
                fprintf('  Converting field: %s\n', field);
            end
            data.(field) = convert_field(data.(field));
            modified = true;
        end
    end
    
    % Save the modified data
    if modified
        save(file_path, '-struct', 'data', '-v7.3');
        if p.Results.Verbose
            fprintf('  Saved updated file: %s\n', file_path);
        end
    else
        if p.Results.Verbose
            fprintf('  No modifications made to: %s\n', file_path);
        end
    end
end

if p.Results.Verbose
    fprintf('Processing complete.\n');
end
end

function trials = convert_trials(trials, verbose)
% Convert trial data structure
if ~isstruct(trials) || ~isscalar(trials)
    if verbose
        fprintf('    Not a scalar structure, skipping trial conversion\n');
    end
    return;
end

fields = fieldnames(trials);
for i = 1:length(fields)
    field = fields{i};
    
    % Skip already converted fields
    if endsWith(field, '_str')
        continue;
    end
    
    % Convert field values
    for j = 1:length(trials)
        if isfield(trials, field)
            trials(j).([field, '_str']) = convert_value(trials(j).(field));
            if verbose && j == 1  % Only show message for first trial
                fprintf('    Converted field: %s -> %s_str\n', field, field);
            end
        end
    end
end
end

function out = convert_field(value)
% Convert a single field value
if isstruct(value)
    out = convert_struct(value);
elseif iscell(value)
    out = convert_cell(value);
else
    out = convert_value(value);
end
end

function out = convert_struct(s, verbose)
% Convert structure fields
out = s;
fields = fieldnames(s);
for i = 1:length(fields)
    field = fields{i};
    if ~endsWith(field, '_str')  % Skip already converted fields
        out.([field, '_str']) = convert_value(s.(field));
        if verbose
            fprintf('    Converted struct field: %s -> %s_str\n', field, field);
        end
    end
end
end

function out = convert_cell(c)
% Convert cell array elements
out = cell(size(c));
for i = 1:numel(c)
    if iscell(c{i})
        out{i} = convert_cell(c{i});
    elseif isstruct(c{i})
        out{i} = convert_struct(c{i});
    else
        out{i} = convert_value(c{i});
    end
end
end

function str = convert_value(value)
% Convert a single value to string representation
if ischar(value) && isrow(value)
    str = value;
elseif isstring(value)
    str = char(value);
elseif isnumeric(value) || islogical(value)
    if isscalar(value)
        str = num2str(value);
    else
        str = mat2str(value);
    end
elseif isa(value, 'function_handle')
    str = func2str(value);
elseif isobject(value)
    try
        str = char(string(value));
    catch
        str = ['[', class(value), ' object]'];
    end
else
    str = ['[', class(value), ']'];
end
end
