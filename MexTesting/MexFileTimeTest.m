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
nCases = 5;


durationVec = zeros(nCases,numTimeSteps);
CaseSensorTemps = ones(nCases,size(sensorPositions,1),numTimeSteps+1)*20;

%% CASE 1 - Repeated calls to MEX with rebuilding matrices
caseNum = 1;
createMatrices = true;
TPrediction = T0;
for i = 1:numTimeSteps
    tic
    [TPrediction,sensorTemps] = MEX_Heat_Simulation(TPrediction,fluenceRate,tissueSize',tFinal,...
        deltaT,tissueProperties,BC,flux,ambientTemp,sensorPositions,layerInfo,...
        useAllCPUs,useGPU,alpha,silentMode,Nn1d,createMatrices);
    CaseSensorTemps(caseNum,:,i+1) = sensorTemps(:,end);
    durationVec(caseNum,i) = toc;
end
fprintf("CASE %d: Total Time %0.3f s\n", caseNum, sum(durationVec(caseNum,:)));

%% CASE 2 - Repeated calls to MEX without rebuilding matrices
caseNum = 2;
createMatrices = true; % create Matrices has to be true for first call to set proper params
TPrediction = T0;
for i = 1:numTimeSteps
    tic
    [TPrediction,sensorTemps] = MEX_Heat_Simulation(TPrediction,fluenceRate,tissueSize',tFinal,...
        deltaT,tissueProperties,BC,flux,ambientTemp,sensorPositions,layerInfo,...
        useAllCPUs,useGPU,alpha,silentMode,Nn1d,createMatrices);
    createMatrices = false; % after first call it will always be false
    CaseSensorTemps(caseNum,:,i+1) = sensorTemps(:,end);
    durationVec(caseNum,i) = toc;
end
fprintf("CASE %d: Total Time %0.3f s\n",caseNum, sum(durationVec(caseNum,:)));
%% CASE 3 - Single call to MEX 
caseNum = 3;
createMatrices = true;
tFinal = numTimeSteps*deltaT;
TPrediction = T0;
tic
[TPrediction,sensorTemps] = MEX_Heat_Simulation(T0,fluenceRate,tissueSize',tFinal,...
    deltaT,tissueProperties,BC,flux,ambientTemp,sensorPositions,layerInfo,...
    useAllCPUs,useGPU,alpha,silentMode,Nn1d,createMatrices);
CaseSensorTemps(caseNum,:,:) = sensorTemps;
durationVec(caseNum,end) = toc;
fprintf("CASE %d: Total Time %0.3f s\n", caseNum, sum(durationVec(caseNum,:)));

%% CASE 4 - Single call to MEX multiStep 
% Note that the point of multistep is to handle changing inputs without
% multiple calls to mex, so it may not be the fastest, but it will be
% faster than anything calling MEX multiple times. 

caseNum = 4;
time = 0:deltaT:deltaT*numTimeSteps; % this will be numTimeSteps + 1 long
laserPose = [0;0;-focalPoint;0;0;0].*ones(6,length(time));
laserPower = ones(1,length(time));
tic
[~,sensorTemps] = MEX_Heat_Simulation_MultiStep(T0,tissueSize',...
    tissueProperties,BC,flux,ambientTemp,sensorPositions,w0,time,...
    laserPose,laserPower,layerInfo,useAllCPUs,useGPU,alpha,...
    silentMode,Nn1d);
CaseSensorTemps(caseNum,:,:) = sensorTemps;
durationVec(caseNum,end) = toc;
fprintf("CASE %d: Total Time %0.3f s\n", caseNum, sum(durationVec(caseNum,:)));

%% CASE 5 - Single call to MEX multiStep but backwards-euler 
% Note that the point of multistep is to handle changing inputs without
% multiple calls to mex, so it may not be the fastest, but it will be
% faster than anything calling MEX multiple times. 

caseNum = 5;
createMatrices = true;
time = 0:deltaT:deltaT*numTimeSteps; % this will be numTimeSteps + 1 long
laserPose = [0;0;-focalPoint;0;0;0].*ones(6,length(time));
laserPower = ones(1,length(time));
TPrediction = T0;
alpha = 1;
tic
[~,sensorTemps] = MEX_Heat_Simulation_MultiStep(T0,tissueSize',...
    tissueProperties,BC,flux,ambientTemp,sensorPositions,w0,time,...
    laserPose,laserPower,layerInfo,useAllCPUs,useGPU,alpha,...
    silentMode,Nn1d);
CaseSensorTemps(caseNum,:,:) = sensorTemps;
durationVec(caseNum,end) = toc;
fprintf("CASE %d: Total Time %0.3f s\n", caseNum, sum(durationVec(caseNum,:)));


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