function traces = readTraces(md)
%
%
%

%%
    traces = csvread(fullfile(md.Location,'SortedICs.csv')); 
    
    traces(:,1)= [];
    traces = traces'; 
    
end