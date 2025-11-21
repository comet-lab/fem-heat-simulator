function [allSensorTemps, allDurations] = runMexTimeTest(useAllCPUs,useGPU,alpha,opts)
arguments
    useAllCPUs (1,1) logical = false
    useGPU (1,1) logical = false
    alpha (1,1) double {mustBeInRange(alpha,0,1)} = 0.5
    opts.silentMode = true;
    opts.numTimeSteps (1,1) double = 20;
    opts.deltaT (1,1) double = 0.05;
    opts.tissueSize (1,3) double = [5,5,1]
    opts.nodesPerAxis (1,3) double {mustBeInteger(opts.nodesPerAxis)} = [101,101,100];
    opts.layerInfo = [0.05, 30];
    opts.filename = '';
end

silentMode = opts.silentMode;
numTimeSteps = opts.numTimeSteps;
deltaT = opts.deltaT;

tissueSize = opts.tissueSize;
nodesPerAxis = opts.nodesPerAxis;
ambientTemp = 24;
T0 = single(20*ones(nodesPerAxis));
tFinal = single(0.05);
w0 = 0.0168;
focalPoint = 35;
MUA = 200;
TC = 0.0062;
VHC = 4.3;
HTC = 0.05;
Nn1d = 2;
layerInfo = opts.layerInfo;
sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.1];

w = @(z) w0 * sqrt(1 + (z.*10.6e-4./(pi*w0.^2)).^2);
I = @(x,y,z,MUA) 2./(w(focalPoint + z).^2.*pi) .* exp(-2.*(x.^2 + y.^2)./(w(focalPoint + z).^2) - MUA.*z);

xLayer = linspace(-tissueSize(1)/2,tissueSize(1)/2,nodesPerAxis(1));
yLayer = linspace(-tissueSize(2)/2,tissueSize(2)/2,nodesPerAxis(2));
zLayer = [linspace(0,layerInfo(1)-layerInfo(1)/layerInfo(2),layerInfo(2)) linspace(layerInfo(1),tissueSize(3),nodesPerAxis(3)-layerInfo(2))];
[Y,X,Z] = meshgrid(yLayer,xLayer,zLayer); % left handed
fluenceRate = single(I(X,Y,Z,MUA));
tissueProperties = [MUA,TC,VHC,HTC]';

BC = int32([2,0,2,2,2,2]'); %0: HeatSink, 1: Flux, 2: Convection
flux = 0;

%% Timing Tests
nCases = 4;

allDurations = zeros(nCases,numTimeSteps);
allSensorTemps = ones(nCases,size(sensorPositions,1),numTimeSteps+1)*20;

%% CASE 1 - Repeated calls to MEX with rebuilding matrices
caseNum = 1;
createMatrices = true;
TPrediction = T0;
for i = 1:numTimeSteps
    tic
    [TPrediction,sensorTemps] = MEX_Heat_Simulation(TPrediction,fluenceRate,tissueSize',tFinal,...
        deltaT,tissueProperties,BC,flux,ambientTemp,sensorPositions,layerInfo,...
        useAllCPUs,useGPU,alpha,silentMode,Nn1d,createMatrices);
    allSensorTemps(caseNum,:,i+1) = sensorTemps(:,end);
    allDurations(caseNum,i) = toc;
end
fprintf("CASE %d: Total Time %0.3f s\n", caseNum, sum(allDurations(caseNum,:)));

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
    allSensorTemps(caseNum,:,i+1) = sensorTemps(:,end);
    allDurations(caseNum,i) = toc;
end
fprintf("CASE %d: Total Time %0.3f s\n",caseNum, sum(allDurations(caseNum,:)));
%% CASE 3 - Single call to MEX 
caseNum = 3;
createMatrices = true;
tFinal = numTimeSteps*deltaT;
TPrediction = T0;
tic
[TPrediction,sensorTemps] = MEX_Heat_Simulation(T0,fluenceRate,tissueSize',tFinal,...
    deltaT,tissueProperties,BC,flux,ambientTemp,sensorPositions,layerInfo,...
    useAllCPUs,useGPU,alpha,silentMode,Nn1d,createMatrices);
allSensorTemps(caseNum,:,:) = sensorTemps;
allDurations(caseNum,end) = toc;
fprintf("CASE %d: Total Time %0.3f s\n", caseNum, sum(allDurations(caseNum,:)));

clear MEX_Heat_Simulation
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
allSensorTemps(caseNum,:,:) = sensorTemps;
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

