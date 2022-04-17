% sampling freq
fs=30000.0;
% create spikes structure
spikes = struct();


% **************** preprocessing *****************************************
% find indices of good clusters
cluster_info = tdfread('cluster_info.tsv');
ind = cluster_info.group == 'good ';
ind = ind(:,1);
spikes.cluID = cluster_info.cluster_id(ind)';
spikes.chan = cluster_info.ch(ind)';
shank = ones([1 length(spikes.chan)]);
ind = spikes.chan>=128;
shank(1,ind) = 2;
spikes.shankID = shank;

% get timestamps for each good cluster
spiketimes = double(readNPY('spike_times.npy'));
clusterID = readNPY('spike_clusters.npy');
spikes.times = {};
for i=1:length(spikes.cluID)
    cluid = spikes.cluID(i);
    idx = clusterID==cluid;
    spikes.times{i} = spiketimes(idx)/fs;
end
spikes.spindices = generateSpinDices(spikes.times);

clear shank spiketimes clusterID cluster_info idx ind i cluid


% **************** ACG/ CCG *****************************************

% calculat the CCG between all the pairs of cells
binSize = 0.001; % 1ms bin size
duration = 0.1; % -50ms:50ms window
[ccg,t] = CCG(spikes.spindices(:,1),spikes.spindices(:,2),'binSize',binSize,'duration',duration);

% plot ACGs and CCGs
figure, 
% Plotting the autocorrelogram (ACG) of the eight cell
subplot(2,1,1)
plot(t,ccg(:,8,8)), title('ASCG'), xlabel('Time (seconds)'), ylabel('Count')
% Plotting the cross correlogram (CCG) between a pair of cells
subplot(2,1,2)
plot(t,ccg(:,1,3)), title('CCG'), xlabel('Time (seconds)'), ylabel('Count')


% ***************** Monosynaptic connection ******************************

% run monosynaptic connections detection
mono_res = ce_MonoSynConvClick (spikes,'includeInhibitoryConnections',true);
gui_MonoSyn(mono_res); % Shows the GUI for manual curation

% Loading preferences
preferences = preferences_ProcessCellMetrics(session);

% make connectivity graph
% cellTypes = unique(cell_metrics.putativeCellType,'stable');
% clusClas = ones(1,length(cell_metrics.putativeCellType));
% for i = 1:length(cellTypes)
%     clusClas(strcmp(cell_metrics.putativeCellType,cellTypes{i}))=i;
% end

% Getting connection paris and cells with connections
putativeConnections = mono_res.sig_con_excitatory;
putativeConnections_inh = mono_res.sig_con_inhibitory;
[cellSubset,~,pairsSubset] = unique([putativeConnections;putativeConnections_inh]);
pairsSubset = reshape(pairsSubset,size([putativeConnections;putativeConnections_inh]));

% Generating connectivity matrix (A)
A = zeros(length(cellSubset),length(cellSubset));
for i = 1:size(putativeConnections,1)
    A(pairsSubset(i,1),pairsSubset(i,2)) = 1;
end
for i = size(putativeConnections,1)+1:size(pairsSubset,1)
    A(pairsSubset(i,1),pairsSubset(i,2)) = 2;
end

% Plotting connectivity matrix
figure, subplot(1,2,1)
imagesc(A), title('Connectivity matrix')

% add color
celltype = cell_metrics.putativeCellType;
nodecolor = zeros(length(celltype),3);
for c=1:length(celltype)  
    if strcmp(celltype{c},'Pyramidal Cell')
        nodecolor(c,:) = [1, 0, 0];
    elseif strcmp(celltype{c},'Narrow Interneuron')
        nodecolor(c,:) = [0, 0, 1];
    else
        nodecolor(c,:) = [0, 1, 1];
    end
end

% Plotting connectivity graph
connectivityGraph = digraph(A);
subplot(1,2,2)
connectivityGraph_plot = plot(connectivityGraph,'Layout','force','Iterations',15, ...
    'NodeColor',nodecolor(cellSubset,:),'MarkerSize',3,'EdgeCData',connectivityGraph.Edges.Weight,...
    'HitTest','off', 'EdgeColor',[0.2 0.2 0.2],'NodeLabel',cellSubset);
title('Connectivity graph');

% find number of connections per cell
conn_percell = zeros(length(celltype),1);
[GC,GR] = groupcounts(pairsSubset(:,1));
conn_percell(GR) = GC;