% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Tutorial for running CellExplorer on your own data from a basepath
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

%  1. Define the basepath of the dataset to run. The dataset should at minimum consist of the raw data and spike sorted data.
basepath = 'E:\VR9_AL\kilosort3'; % Z:\peterp03\IntanData\MS21\Peter_MS21_180627_143449_concat
cd(basepath)

%% 2. Generate session metadata struct using the template function and display the meta data in a gui
session = sessionTemplate(pwd,'showGUI',true);

% You can inspect the session struct with the gui 
session = gui_session(session);
% And verify the required and optional fields
verifySessionStruct(session);

%% 3. Run the cell metrics pipeline 'ProcessCellMetrics' using the session struct as input
cell_metrics = ProcessCellMetrics('session', session);

%% 4. Visualize the cell metrics in CellExplorer
cell_metrics = CellExplorer('metrics',cell_metrics); 
