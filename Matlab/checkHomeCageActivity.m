function checkHomeCageActivity(md)
%
%
%

    [fearCells,events,t] = getFearBoxCells(md);
    transients = events>0;
    inBox = find(t==500):find(t==t(end)-500);
    
    firstHC = transients(fearCells,1:inBox(1)-1);
    secondHC = transients(fearCells,(inBox(end)+1):end);
    
    firstHC = sum(firstHC,2);
    secondHC = sum(secondHC,2);
    
    errorbar([1,2],[mean(firstHC),mean(secondHC)],...
        [standarderror(firstHC),standarderror(secondHC)]);
    xlim([.5 2.5]);
end