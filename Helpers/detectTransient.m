function detectTransient(md)
%
%
%

%%
    traces = csvread(fullfile(md.Location,'SortedICs.csv')); 
    traces = traces';
    traces(1,:) = []; 
    nNeurons = size(traces,1); 
    dfdt = [zeros(nNeurons,1) diff(traces,[],2)];
    keyboard; 
end