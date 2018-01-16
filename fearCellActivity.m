function eventsInThisBlock = fearCellActivity(md)
%
%
%

    [fearCells,events,t] = getFearBoxCells(md);
    nNeurons = size(events,1);
    nonFearCells = setdiff(1:nNeurons,fearCells); 
    transients = events>0;
    nFearCells = length(fearCells); 
    
    putInBox = find(t==500)-1; 
    blockLims = putInBox;
    for b=1:6
        blockLims = [blockLims blockLims(b)+6000];
    end
    
    eventsInThisBlock = nan(nFearCells,6);
    eventsInThisBlock_nonfear = nan(length(nonFearCells),6);
    for b=1:6
        start = blockLims(b)+1;
        stop = blockLims(b+1); 
        eventsInThisBlock(:,b) = sum(transients(fearCells,start:stop),2);
        
        eventsInThisBlock_nonfear(:,b) = sum(transients(nonFearCells,start:stop),2);
    end
    
    figure; hold on;
    errorbar([1:6],mean(eventsInThisBlock),standarderror(eventsInThisBlock));
    errorbar([1:6],mean(eventsInThisBlock_nonfear),standarderror(eventsInThisBlock_nonfear));
    ylabel('# calcium transients');
    xlabel('Time (5 min. bins)');
    set(gca,'xtick',[1 2 3 4 5 6]);
    xlim([.5 6.5])
    make_plot_pretty(gca);
    legend({'Fear cells','Non fear cells'});

end