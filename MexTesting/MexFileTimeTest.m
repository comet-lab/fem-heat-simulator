clc; clear; close all;  

clear MEX_Heat_Simulation
clear MEX_Heat_Simulation_MultiStep
%% Initialization of parameters
useAllCPUs = false;
useGPU = true;
silentMode = true;

numTimeSteps = 20;
deltaT = 0.05;

tissueSize = [5.0,5.0,1.0];
nodesPerAxis = [101,101,100];
ambientTemp = 24;
T0 = single(20*ones(nodesPerAxis));
alpha = 1/2;
tFinal = single(0.05);
w0 = 0.0168;
focalPoint = 35;
MUA = 200;
TC = 0.0062;
VHC = 4.3;
HTC = 0.05;
Nn1d = 2;
layerInfo = [0.05,30];
sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.1];

w = @(z) w0 * sqrt(1 + (z.*10.6e-4./(pi*w0.^2)).^2);
I = @(x,y,z,MUA) 2./(w(focalPoint + z).^2.*pi) .* exp(-2.*(x.^2 + y.^2)./(w(focalPoint + z).^2) - MUA.*z);

xLayer = linspace(-tissueSize(1)/2,tissueSize(1)/2,nodesPerAxis(1));
yLayer = linspace(-tissueSize(2)/2,tissueSize(2)/2,nodesPerAxis(2));
zLayer = [linspace(0,layerInfo(1)-layerInfo(1)/layerInfo(2),layerInfo(2)) linspace(layerInfo(1),tissueSize(3),nodesPerAxis(3)-layerInfo(2))];
[Y,X,Z] = meshgrid(yLayer,xLayer,zLayer); % left handed
fluenceRate = single(I(X,Y,Z,MUA));
tissueProperties = [MUA,TC,VHC,HTC]';

BC = int32([2,0,0,0,0,0]'); %0: HeatSink, 1: Flux, 2: Convection
flux = 0;

%% Timing Tests
useCPU = false; useGPU = true; alpha = 0.5;
[CaseSensorTemps, durationVec] = runMexTimeTest(useCPU, useGPU, alpha);
nCases = size(CaseSensorTemps,1);

%% Plot Sensor Temps to confirm the different methods produce the same result

time = 0:deltaT:deltaT*numTimeSteps;
figure(1);
clf;
tl = tiledlayout('flow');

for ss = 1:size(sensorPositions,1)
    ax = nexttile();
    for cc = 1:nCases
        hold on;
        plot(time,reshape(CaseSensorTemps(cc,ss,:),size(time)),'LineWidth',2,'DisplayName',...
            sprintf("Case %d",cc));
        hold off
    end
    title(sprintf("Location (%g, %g, %g)",sensorPositions(ss,:)));
    xlabel("Temperature (deg C)");
    ylabel("Time (s)");
    grid on;
end
leg = legend('Orientation','horizontal');
leg.Layout.Tile = 'south';