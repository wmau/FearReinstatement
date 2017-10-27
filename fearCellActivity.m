function fearCellActivity(md)
%
%
%

    [fearCells,events,t] = getFearBoxCells(md);
    transients = events>0;
    nFearCells = length(fearCells); 
    
    putInBox = find(t==500)-1; 
    blockLims = putInBox;
    for b=1:6
        blockLims = [blockLims blockLims(b)+6000];
    end
    
    eventsInThisBlock = nan(nFearCells,6);
    for b=1:6
        start = blockLims(b)+1;
        stop = blockLims(b+1); 
        eventsInThisBlock(:,b) = sum(transients(fearCells,start:stop),2);
    end
    
    errorbar([1:6],mean(eventsInThisBlock),standarderror(eventsInThisBlock));
    ylabel('# calcium transients');
    xlabel('Time (5 min. bins)');
    set(gca,'xtick',[1 2 3 4 5 6]);
    xlim([.5 6.5])
end