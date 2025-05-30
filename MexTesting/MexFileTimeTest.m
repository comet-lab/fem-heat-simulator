clc; clear; close all;

clear MEX_Heat_Simulation
%% Initialization of parameters
tissueSize = [2.0,2.0,1.0];
nodesPerAxis = [41,41,71];
ambientTemp = 24;
T0 = single(20*ones(nodesPerAxis));
deltaT = 0.05;
tFinal = single(0.05);
w0 = 0.0168;
focalPoint = 35;
MUA = 200;
TC = 0.0062;
VHC = 4.3;
HTC = 0.05;
useAllCPUs = true;
silentMode = true;
Nn1d = 2;
layerInfo = [0.05,30];
sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.5; 0,0,0.95; 0 0 1];

w = @(z) w0 * sqrt(1 + (z.*10.6e-4./(pi*w0.^2)).^2);
I = @(x,y,z,MUA) 2./(w(focalPoint + z).^2.*pi) .* exp(-2.*(x.^2 + y.^2)./(w(focalPoint + z).^2) - MUA.*z);

xLayer = linspace(-tissueSize(1)/2,tissueSize(1)/2,nodesPerAxis(1));
yLayer = linspace(-tissueSize(2)/2,tissueSize(2)/2,nodesPerAxis(2));
zLayer = [linspace(0,layerInfo(1)-layerInfo(1)/layerInfo(2),layerInfo(2)) linspace(layerInfo(1),tissueSize(3),nodesPerAxis(3)-layerInfo(2))];
[X,Y,Z] = meshgrid(xLayer,yLayer,zLayer);
fluenceRate = single(I(X,Y,Z,MUA));
tissueProperties = [MUA,TC,VHC,HTC]';

BC = int32([2,0,0,0,0,0]'); %0: HeatSink, 1: Flux, 2: Convection
Flux = 0;

%% Timing Tests
numSamples = 100;
timeVec = zeros(2,numSamples);
for cc = 0:1
    createMatrices = logical(cc);
    for i = 1:numSamples
        tic
        if ~silentMode
            fprintf("\n");
        end
        [TPrediction,sensorTempsLayer] = MEX_Heat_Simulation(T0,fluenceRate,tissueSize',tFinal,...
            deltaT,tissueProperties,BC,Flux,ambientTemp,sensorPositions,useAllCPUs,...
            silentMode,layerInfo,Nn1d,createMatrices);
        timeVec(cc+1,i) = toc;
    end
    fprintf("CreateMatrices is %d\n", cc);
    fprintf("Average time per run: %0.4f s\n", mean(timeVec(cc+1,:)));
    fprintf("Total time for %d samples: %0.4f\n", numSamples, sum(timeVec(cc+1,:)));
end