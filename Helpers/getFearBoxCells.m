function [fearCells,events,t] = getFearBoxCells(md)
%
%
%

%%
    cd(md.Location);
    events = csvread('SortedEvents.csv'); 
    t = events(:,1);
    events(:,1) = []; 
    events = events';
    transients = events > 0;
    [nNeurons,nFrames] = size(events); 
    
    inBox = find(t==500):find(t==t(end)-500);
    outOfBox = setdiff(1:nFrames,inBox);
    
    inBoxTransients = sum(transients(:,inBox),2)./length(inBox);
    outofBoxTransients = sum(transients(:,outOfBox),2)./length(outOfBox);
    
    fearCells = find(inBoxTransients > outofBoxTransients)';
        
end