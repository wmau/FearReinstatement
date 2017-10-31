session = MD(312); 

traces = readTraces(session);
[fearCells,events,t] = getFearBoxCells(session);
cherryPicked = [28 29 53 26 22 21 18 10 7 67];

figure('Position',[590 240 340 640]);
n = 10;
colors = distinguishable_colors(n);
c = zeros(size(traces,1),3);
c(fearCells(cherryPicked),:) = colors; 
counter = 1;
for i=cherryPicked
    AX(counter) = subplot(10,1,counter);
    plot(t,traces(fearCells(i),:),'color',c(fearCells(i),:));
    ylim([min(traces(fearCells(i),:)), max(traces(fearCells(i),:))]);
    axis off; 
    counter = counter+1; 
end

set(AX,'YLim',[min([AX.YLim]),max([AX.YLim])],...
    'XLim',[0,max(t)]);
axis on;

proj = imread('MaxProj.tif'); 
figure('Position',[950 260 760 620]);
imshow(proj,[]);
caxis([0 1700]);
hold on;
PlotNeurons(session,1:size(traces,1),[0.5843    0.8157    0.9882],1);
PlotNeurons(session,fearCells(cherryPicked)',c,3);
line([0 100*1.10],[5 5],'linewidth',5,'color','w');