clc; clear; close all;
% Initialization of parameters
sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.1; 0,0,0.2; 0 0 0.3;0 0 0.4;0 0 0.5];
z = [0:0.0015:0.05 0.1:0.05:1];
x = linspace(-1,1,41);
y = linspace(-1,1,41);
nodesPerAxis = [length(x),length(y),length(z)];
boundaryConditions = [1,2,1,1,1,1]';

mesh = Mesh(x,y,z,boundaryConditions);
% Initialization of parameters
simulator = HeatSimulator();
simulator.dt = 0.05;
simulator.alpha = 0.5;
simulator.useAllCPUs = false;
simulator.useGPU = false;
simulator.silentMode = true;
simulator.mesh = mesh;
simulator.sensorLocations = sensorPositions;

thermalInfo = ThermalModel();
thermalInfo.MUA = 400;
thermalInfo.TC = 0.0062;
thermalInfo.VHC = 4.3;
thermalInfo.HTC = 0.008;
thermalInfo.ambientTemp = 24;
thermalInfo.flux = 0;
T0 = 20*ones(prod(nodesPerAxis),1);
thermalInfo.temperature = T0;
simulator.thermalInfo = thermalInfo;


% set laser settings
w0 = 0.0168;
lambda = 10.6e-4;
laser = Laser(w0,lambda,thermalInfo.MUA);
laser.focalPose = struct('x',0,'y',0,'z',-35,'theta',0,'phi',0,'psi',0);
laser = laser.calculateIrradiance(mesh);
simulator.laser = laser;

tFinal = 1;
timePoints = (0:simulator.dt:tFinal)';
numTimeSteps = length(timePoints)-1;
nRuns = 10;
% Each loop will run with its own version of muA cv and T
muAOpts = linspace(2,200,nRuns);
cvOpts = linspace(3.0,4.5,nRuns);
Tstart = cell(nRuns,1);
for i = 1:nRuns
    Tstart{i} = T0;
end

%% Parallel loop with matrix rebuild On;
myCluster = parcluster('Processes');

bigTic = tic;
parDurationVec = zeros(nRuns,numTimeSteps);
parSensorTemps = ones(nRuns,size(sensorPositions,1),numTimeSteps+1)*20;
Tprediction = Tstart;
simulator.buildMatrices = true;
simulator.resetIntegration = true;
for t = 1:numTimeSteps
    stepSensorTemps = zeros(nRuns,size(sensorPositions,1));
    parfor i = 1:nRuns
        simCopy = simulator.deepCopy();
        simCopy.thermalInfo.MUA = muAOpts(i);
        simCopy.thermalInfo.VHC = cvOpts(i);
        simCopy.thermalInfo.temperature = Tprediction{i};
        tic
        [Tpred,sensorTemps] = simCopy.solve([0;simCopy.dt]);
        stepSensorTemps(i,:) = sensorTemps(end,:);
        parDurationVec(i,t) = toc;
        Tprediction{i} = Tpred;
    end
    parSensorTemps(:,:,t+1) = stepSensorTemps;
end
toc(bigTic)
%% Parallel loop with buildMatrices OFF but resetIntegration true;
clear MEX_Heat_Simulation;

bigTic = tic;
parOffDurationVec = zeros(nRuns,numTimeSteps);
parOffSensorTemps = ones(nRuns,size(sensorPositions,1),numTimeSteps+1)*20;
Tprediction = Tstart;
simulator.buildMatrices = true;
simulator.resetIntegration = true;
for t = 1:numTimeSteps
    stepSensorTemps = zeros(nRuns,size(sensorPositions,1));
    parfor i = 1:nRuns
        simCopy = simulator.deepCopy();
        simCopy.thermalInfo.MUA = muAOpts(i);
        simCopy.thermalInfo.VHC = cvOpts(i);
        simCopy.thermalInfo.temperature = Tprediction{i};
        tic
        [Tpred,sensorTemps] = simCopy.solve([0;simCopy.dt]);
        stepSensorTemps(i,:) = sensorTemps(end,:);
        parOffDurationVec(i,t) = toc;
        Tprediction{i} = Tpred;
    end
    parOffSensorTemps(:,:,t+1) = stepSensorTemps;
    simulator.buildMatrices = false;
    simulator.resetIntegration = true;
end
toc(bigTic)

%% Parallel loop with buildMatrices OFF and resetIntegration OFF;
clear MEX_Heat_Simulation;

bigTic = tic;
parOffOffDurVec = zeros(nRuns,numTimeSteps);
parOffOffSensTemps = ones(nRuns,size(sensorPositions,1),numTimeSteps+1)*20;
Tprediction = Tstart;
simulator.buildMatrices = true;
simulator.resetIntegration = true;
for t = 1:numTimeSteps
    stepSensorTemps = zeros(nRuns,size(sensorPositions,1));
    parfor i = 1:nRuns
        simCopy = simulator.deepCopy();
        simCopy.thermalInfo.MUA = muAOpts(i);
        simCopy.thermalInfo.VHC = cvOpts(i);
        simCopy.thermalInfo.temperature = Tprediction{i};
        tic
        [Tpred,sensorTemps] = simCopy.solve([0;simCopy.dt]);
        stepSensorTemps(i,:) = sensorTemps(end,:);
        parOffOffDurVec(i,t) = toc;
        Tprediction{i} = Tpred;
    end
    parOffOffSensTemps(:,:,t+1) = stepSensorTemps;
    simulator.buildMatrices = false;
    simulator.resetIntegration = false;
end
toc(bigTic)
%% Single CPU Loop with rebuild off 
clear MEX_Heat_Simulation;

bigTic = tic;
singOnDurationVec = zeros(nRuns,numTimeSteps);
singOnSensorTemps = ones(nRuns,size(sensorPositions,1),numTimeSteps+1)*20;
Tprediction = Tstart;
simulator.buildMatrices = true;
simulator.resetIntegration = true;
for t = 1:numTimeSteps
    stepSensorTemps = zeros(nRuns,size(sensorPositions,1));
    for i = 1:nRuns
        simulator.thermalInfo.MUA = muAOpts(i);
        simulator.thermalInfo.VHC = cvOpts(i);
        simulator.thermalInfo.temperature = Tprediction{i};
        tic
        [Tpred,sensorTemps] = simulator.solve([0;simulator.dt]);
        Tprediction{i} = Tpred;

        stepSensorTemps(i,:) = sensorTemps(end,:);
        singOnDurationVec(i,t) = toc;
        simulator.buildMatrices = false;
        simulator.resetIntegration = true;
    end
    singOnSensorTemps(:,:,t+1) = stepSensorTemps;
end
toc(bigTic)


%% Ground Truth
clear MEX_Heat_Simulation;

bigTic = tic;
singDurationVec = zeros(nRuns,numTimeSteps);
singSensorTemps = ones(nRuns,size(sensorPositions,1),numTimeSteps+1)*20;
Tprediction = Tstart;
simulator.buildMatrices = true;
simulator.resetIntegration = true;
for i = 1:nRuns
    stepSensorTemps = zeros(size(sensorPositions,1),numTimeSteps);
    simulator.buildMatrices = true;
    simulator.resetIntegration = true;
    simulator.thermalInfo.MUA = muAOpts(i);
    simulator.thermalInfo.VHC = cvOpts(i);
    simulator.thermalInfo.temperature = Tprediction{i};
    for t = 1:numTimeSteps

        tic
        [Tpred,sensorTemps] = simulator.solve([0;simulator.dt]);
        stepSensorTemps(:,t) = sensorTemps(end,:)';
        singDurationVec(i,t) = toc;
        simulator.buildMatrices = false;
        simulator.resetIntegration = false;
    end
    Tprediction{i} = Tpred;
    singSensorTemps(i,:,2:end) = stepSensorTemps;
end
toc(bigTic)

%% PLotting
figure(1);
clf;
tl = tiledlayout('flow');

time = (0:simulator.dt:tFinal)';
for i = 1:nRuns
nexttile()
hold on;
plot(time,reshape(singOnSensorTemps(i,1,:),size(time)),'Linewidth',2,'DisplayName',"Single-Off-On")
plot(time,reshape(singSensorTemps(i,1,:),size(time)),'Linewidth',2,'DisplayName',"Ground Truth")
plot(time,reshape(parSensorTemps(i,1,:),size(time)),'--','Linewidth',2,'DisplayName',"Parallel-On-On")
plot(time,reshape(parOffSensorTemps(i,1,:),size(time)),'-.','Linewidth',2,'DisplayName',"Parallel-Off-On")
plot(time,reshape(parOffOffSensTemps(i,1,:),size(time)),':','Linewidth',2,'DisplayName',"Parallel-Off-Off")
hold off
grid on;
xlabel("Time (s)");
ylabel("Temperature (deg C)");
title(sprintf("Run %d", i));

end
leg = legend('orientation','horizontal');
leg.Layout.Tile = 'south';
clear mex;

