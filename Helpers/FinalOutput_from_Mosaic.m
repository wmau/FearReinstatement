function FinalOutput_from_Mosaic(md)
%
%
%

%%  
    currDir = pwd; 
    
    %Go to the directory and grab the file names. 
    cd(fullfile(md.Location,'SortedROIs'));
    
    %Load an example cell to get imaging dimensions. 
    tifs = natsortfiles(cellstr(ls('*.tif'))); 
    tifs = natsortfiles(cellstr(ls('*.tif'))); 
    nNeurons = length(tifs);
    
%%
    [NeuronImage,NeuronAvg] = deal(cell(1,nNeurons));
    for neuron = 1:nNeurons
        mask = imread(tifs{neuron}); 
        NeuronImage{neuron} = mask > (max(mask(:))/2);
        NeuronAvg{neuron} = mask(mask > (max(mask(:))/2)); 
    end

    NumNeurons = length(NeuronImage);
    
    cd(md.Location); 
    save('FinalOutput.mat','NeuronAvg','NeuronImage','NumNeurons');
    cd(currDir); 
end