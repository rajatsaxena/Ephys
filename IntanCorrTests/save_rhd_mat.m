% get the name of files ending with .rhd format
files = dir('*.rhd');

% iterate over each file in the list of files from above
for i=1:length(files)
    filename = files(i).name;
    
    % process the intan data into matlab formats; header file available
    % online at intan website (if you want to know more
    [amp_data, ~, freq_params] = read_Intan_RHD2000_file(filename);
    % load the sampling rate 
    fs = freq_params.board_adc_sample_rate;
    amp_data = single(amp_data);
    
    % mat file name used to store mat data
    matfilename = strsplit(filename,'.rhd');
    matfilename = matfilename{1};
    matfilename = strcat(matfilename,'.mat');
    
    disp('Saving to mat file');
    save(matfilename, 'amp_data', 'fs', '-v7.3');
    
    % clean data for better loading
    clear amp_data amp_ts amp_channels freq_params
    
end
