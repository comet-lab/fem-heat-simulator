function [allSensorTemps, allDurations] = runMexTimeTest(useAllCPUs,useGPU,alpha,opts)
arguments
    useAllCPUs (1,1) logical = false
    useGPU (1,1) logical = false
    alpha (1,1) double {mustBeInRange(alpha,0,1)} = 0.5
    opts.silentMode = true;
    opts.numTimeSteps (1,1) double = 20;
    opts.deltaT (1,1) double = 0.05;
    opts.nodes (:,3) double = [];
    opts.filename = '';
end

numTimeSteps = opts.numTimeSteps;

sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.1; 0,0,0.2; 0 0 0.3;0 0 0.4;0 0 0.5];
if isempty(opts.nodes)
    z = [0:0.0015:0.05 0.1:0.05:1];
    x = linspace(-2.5,2.5,101);
    y = linspace(-2.5,2.5,101);
else
    x = opts.nodes(:,1);
    y = opts.nodes(:,2);
    z = opts.nodes(:,3);
end
nodesPerAxis = [length(x),length(y),length(z)];
boundaryConditions = [1,2,1,1,1,1]';

mesh = Mesh(x,y,z,boundaryConditions);
% Initialization of parameters
simulator = HeatSimulator();
simulator.dt = opts.deltaT;
simulator.alpha = alpha;
simulator.useAllCPUs = useAllCPUs;
simulator.useGPU = useGPU;
simulator.silentMode = opts.silentMode;
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

% set laser parameters
w0 = 0.0168;
lambda = 10.6e-4;
focalPoint = 35;
laser = Laser(w0,lambda,thermalInfo.MUA);
laser.focalPose = struct('x',0,'y',0,'z',-focalPoint,'theta',0,'phi',0,'psi',0);
laser = laser.calculateIrradiance(mesh);
simulator.laser = laser;


%% Timing Tests
nCases = 4;

allDurations = zeros(nCases,numTimeSteps);
allSensorTemps = ones(nCases,size(sensorPositions,1),numTimeSteps+1)*20;
timePoints = [0;simulator.dt];

%% CASE 1 - Repeated calls to MEX with rebuilding matrices
caseNum = 1;
simulator.buildMatrices = true;
for i = 1:numTimeSteps
    tic
    [~,sensorTemps] = simulator.solve(timePoints);
    allSensorTemps(caseNum,:,i+1) = sensorTemps(end,:)';
    allDurations(caseNum,i) = toc;
end
fprintf("CASE %d: Total Time %0.3f s\n", caseNum, sum(allDurations(caseNum,:)));

%% CASE 2 - Repeated calls to MEX without rebuilding matrices
caseNum = 2;
simulator.buildMatrices = true; % create Matrices has to be true for first call to set proper params
simulator.thermalInfo.temperature = T0;
for i = 1:numTimeSteps
    tic
    [~,sensorTemps] = simulator.solve(timePoints);
    simulator.buildMatrices = false; % after first call it will always be false
    simulator.resetIntegration = false; % after first call it will always be false
    allSensorTemps(caseNum,:,i+1) = sensorTemps(end,:)';
    allDurations(caseNum,i) = toc;
end
fprintf("CASE %d: Total Time %0.3f s\n",caseNum, sum(allDurations(caseNum,:)));
%% CASE 3 - Single call to MEX 
caseNum = 3;
simulator.buildMatrices = true;
simulator.resetIntegration = true;
simulator.thermalInfo.temperature = T0;
timePoints = (0:simulator.dt:opts.numTimeSteps*simulator.dt)';
tic
[~,sensorTemps] = simulator.solve(timePoints);
allSensorTemps(caseNum,:,:) = sensorTemps';
allDurations(caseNum,end) = toc;
fprintf("CASE %d: Total Time %0.3f s\n", caseNum, sum(allDurations(caseNum,:)));

clear MEX_Heat_Simulation
%% CASE 4 - Single call to MEX will changing laser settings 
% Note that the point of multistep is to handle changing inputs without
% multiple calls to mex, so it may not be the fastest, but it will be
% faster than anything calling MEX multiple times. 

caseNum = 4;
time = (0:simulator.dt:opts.numTimeSteps*simulator.dt)'; % this will be numTimeSteps + 1 long
laserPose = [0, 0, -focalPoint, 0, 0, 0].*ones(length(time),6);
laserPower = ones(length(time),1);
simulator.thermalInfo.temperature = T0;
tic
[~,sensorTemps] = simulator.solve(timePoints,laserPose,laserPower);
allSensorTemps(caseNum,:,:) = sensorTemps';
allDurations(caseNum,end) = toc;
fprintf("CASE %d: Total Time %0.3f s\n", caseNum, sum(allDurations(caseNum,:)));

if ~isempty(opts.filename)
    rowHeader = {'useGPU','parallelCPU','alpha','Case1','Case2','Case3','Case4'};
    if ~isfile(opts.filename)
        writecell(rowHeader,opts.filename,'Delimiter','tab','WriteMode','Append');
    end
    data = {useGPU,useAllCPUs,alpha,sum(allDurations(1,:)),sum(allDurations(2,:)),sum(allDurations(3,:)),sum(allDurations(4,:))};
    writecell(data,opts.filename,'Delimiter','tab','WriteMode','Append');
end

clear mex

end

