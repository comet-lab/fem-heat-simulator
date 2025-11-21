clc; clear; close all;
%% Initialization of parameters
tissueSize = [2.0,2.0,1.0];
nodesPerAxis = [41,41,50];
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
useAllCPUs = false;
useGPU = false;
silentMode = true;
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
rng(1);

numTimeSteps = 20;
nRuns = 10;
% Each loop will run with its own version of muA;
muAOpts = linspace(2,200,nRuns);
cvOpts = linspace(3.0,4.5,nRuns);
Tstart = cell(nRuns,1);
for i = 1:nRuns
    Tstart{i} = T0 + randn(size(T0));
end

%% Parallel loop with createMatrices On;
% useGPU = true;
bigTic = tic;
parDurationVec = zeros(nRuns,numTimeSteps);
parSensorTemps = ones(nRuns,size(sensorPositions,1),numTimeSteps+1)*20;
Tprediction = Tstart;
createMatrices = true;
for t = 1:numTimeSteps
    stepSensorTemps = zeros(nRuns,size(sensorPositions,1));
    parfor i = 1:nRuns
        tissuePropTemp = tissueProperties;
        tissuePropTemp(1) = muAOpts(i);
        tissuePropTemp(3) = cvOpts(i);
        Tpred = Tprediction{i};
        tic
        [Tpred,sensorTemps] = MEX_Heat_Simulation(Tpred,fluenceRate,tissueSize',tFinal,...
            deltaT,tissuePropTemp,BC,flux,ambientTemp,sensorPositions,layerInfo,...
            useAllCPUs,useGPU,alpha,silentMode,Nn1d,createMatrices);
        stepSensorTemps(i,:) = sensorTemps(:,end)';
        parDurationVec(i,t) = toc;
        Tprediction{i} = Tpred;
    end
    parSensorTemps(:,:,t+1) = stepSensorTemps;
end
toc(bigTic)
%% Parallel loop with createMatrices OFF;
clear MEX_Heat_Simulation;

bigTic = tic;
parOffDurationVec = zeros(nRuns,numTimeSteps);
parOffSensorTemps = ones(nRuns,size(sensorPositions,1),numTimeSteps+1)*20;
Tprediction = Tstart;
createMatrices = true;
for t = 1:numTimeSteps
    stepSensorTemps = zeros(nRuns,size(sensorPositions,1));
    parfor i = 1:nRuns
        tissuePropTemp = tissueProperties;
        tissuePropTemp(1) = muAOpts(i);
        tissuePropTemp(3) = cvOpts(i);
        Tpred = Tprediction{i};
        tic
        [Tpred,sensorTemps] = MEX_Heat_Simulation(Tpred,fluenceRate,tissueSize',tFinal,...
            deltaT,tissuePropTemp,BC,flux,ambientTemp,sensorPositions,layerInfo,...
            useAllCPUs,useGPU,alpha,silentMode,Nn1d,createMatrices);
        stepSensorTemps(i,:) = sensorTemps(:,end)';
        parOffDurationVec(i,t) = toc;
        Tprediction{i} = Tpred;
    end
    parOffSensorTemps(:,:,t+1) = stepSensorTemps;
    createMatrices = false;
end
toc(bigTic)
%% Single CPU Loop
clear MEX_Heat_Simulation;

bigTic = tic;
singOnDurationVec = zeros(nRuns,numTimeSteps);
singOnSensorTemps = ones(nRuns,size(sensorPositions,1),numTimeSteps+1)*20;
Tprediction = Tstart;
createMatrices = true;
for t = 1:numTimeSteps
    stepSensorTemps = zeros(nRuns,size(sensorPositions,1));
    for i = 1:nRuns
        tissuePropTemp = tissueProperties;
        tissuePropTemp(1) = muAOpts(i);
        tissuePropTemp(3) = cvOpts(i);
        Tpred = Tprediction{i};
        tic
        [Tpred,sensorTemps] = MEX_Heat_Simulation(Tpred,fluenceRate,tissueSize',tFinal,...
            deltaT,tissuePropTemp,BC,flux,ambientTemp,sensorPositions,layerInfo,...
            useAllCPUs,useGPU,alpha,silentMode,Nn1d,createMatrices);
        stepSensorTemps(i,:) = sensorTemps(:,end)';
        singOnDurationVec(i,t) = toc;
        Tprediction{i} = Tpred;
    end
    singOnSensorTemps(:,:,t+1) = stepSensorTemps;
end
toc(bigTic)


%% Single GPU Loop
clear MEX_Heat_Simulation;

useGPU = true;
bigTic = tic;
singDurationVec = zeros(nRuns,numTimeSteps);
singSensorTemps = ones(nRuns,size(sensorPositions,1),numTimeSteps+1)*20;
Tprediction = Tstart;

for i = 1:nRuns
    createMatrices = true;
    stepSensorTemps = zeros(1,size(sensorPositions,1),numTimeSteps);
    tissuePropTemp = tissueProperties;
    tissuePropTemp(1) = muAOpts(i);
    tissuePropTemp(3) = cvOpts(i);
    Tpred = Tprediction{i};
    for t = 1:numTimeSteps
        tic
        [Tpred,sensorTemps] = MEX_Heat_Simulation(Tpred,fluenceRate,tissueSize',tFinal,...
            deltaT,tissuePropTemp,BC,flux,ambientTemp,sensorPositions,layerInfo,...
            useAllCPUs,useGPU,alpha,silentMode,Nn1d,createMatrices);
        stepSensorTemps(1,:,t) = sensorTemps(:,end)';
        singDurationVec(i,t) = toc;
%         createMatrices = false;
    end
    Tprediction{i} = Tpred;
    singSensorTemps(i,:,2:end) = stepSensorTemps;
end
toc(bigTic)

%% PLotting
figure(1);
clf;
tl = tiledlayout('flow');

time = 0:deltaT:numTimeSteps*deltaT;
for i = 1:nRuns
nexttile()
hold on;
plot(time,reshape(singOnSensorTemps(i,1,:),size(time)),'Linewidth',2,'DisplayName',"Single-On")
plot(time,reshape(singSensorTemps(i,1,:),size(time)),'Linewidth',2,'DisplayName',"Single-On GPU")
plot(time,reshape(parSensorTemps(i,1,:),size(time)),'--','Linewidth',2,'DisplayName',"Parallel-On")
plot(time,reshape(parOffSensorTemps(i,1,:),size(time)),':','Linewidth',2,'DisplayName',"Parallel-Off")
hold off
grid on;
xlabel("Time (s)");
ylabel("Temperature (deg C)");
title(sprintf("Run %d", i));
legend();
end

clear mex;

