clc; clear; close all;

clear MEX_Heat_Simulation
%% Initialization of parameters
tissueSize = [2.0,2.0,1.0];
nodesPerAxis = [41,41,71];
ambientTemp = 24;
T0 = single(20*ones(nodesPerAxis));
deltaT = 0.05;
alpha = 1/2;
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
nCases = 3;
numTimeSteps = 100;
timeVec = zeros(nCases,numTimeSteps);
CaseSensorTemps = ones(nCases,size(sensorPositions,1),numTimeSteps+1)*20;

%% CASE 1 - Repeated calls to MEX with rebuilding matrices
caseNum = 1;
createMatrices = true;
TPrediction = T0;
for i = 1:numTimeSteps
    tic
    [TPrediction,sensorTemps] = MEX_Heat_Simulation(TPrediction,fluenceRate,tissueSize',tFinal,...
            deltaT,tissueProperties,BC,Flux,ambientTemp,sensorPositions,useAllCPUs,...
            silentMode,layerInfo,Nn1d,alpha,createMatrices);
    CaseSensorTemps(caseNum,:,i+1) = sensorTemps(:,end);
    timeVec(caseNum,i) = toc;
end
fprintf("CASE %d: Total Time %0.3f s\n", caseNum, sum(timeVec(caseNum,:)));

%% CASE 2 - Repeated calls to MEX without rebuilding matrices
caseNum = 2;
createMatrices = true; % create Matrices has to be true for first call to set proper params
TPrediction = T0;
for i = 1:numTimeSteps
    tic
    [TPrediction,sensorTemps] = MEX_Heat_Simulation(TPrediction,fluenceRate,tissueSize',tFinal,...
            deltaT,tissueProperties,BC,Flux,ambientTemp,sensorPositions,useAllCPUs,...
            silentMode,layerInfo,Nn1d,alpha,createMatrices);
    createMatrices = false; % after first call it will always be false
    CaseSensorTemps(caseNum,:,i+1) = sensorTemps(:,end);
    timeVec(caseNum,i) = toc;
end
fprintf("CASE %d: Total Time %0.3f s\n",caseNum, sum(timeVec(caseNum,:)));
%% CASE 3 - Single call to MEX 
caseNum = 3;
createMatrices = true;
tFinal = numTimeSteps*deltaT;
TPrediction = T0;
tic
[TPrediction,sensorTemps] = MEX_Heat_Simulation(TPrediction,fluenceRate,tissueSize',tFinal,...
        deltaT,tissueProperties,BC,Flux,ambientTemp,sensorPositions,useAllCPUs,...
        silentMode,layerInfo,Nn1d,alpha,createMatrices);
CaseSensorTemps(caseNum,:,:) = sensorTemps;
timeVec(caseNum,end) = toc;
fprintf("CASE %d: Total Time %0.3f s\n", caseNum, sum(timeVec(caseNum,:)));

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