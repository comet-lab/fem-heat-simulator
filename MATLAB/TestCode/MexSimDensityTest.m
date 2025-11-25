clear; close all; clc

sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.5; 0,0,0.95; 0 0 1];

% Initialization of parameters
simulator = HeatSimulator();
simulator.dt = 0.05;
simulator.alpha = 0.5;
simulator.useAllCPUs = false;
simulator.useGPU = false;
simulator.silentMode = false;

simulator.sensorLocations = sensorPositions;

thermalInfo = ThermalModel();
thermalInfo.MUA = 200;
thermalInfo.TC = 0.0062;
thermalInfo.VHC = 4.3;
thermalInfo.HTC = 0.01;
thermalInfo.ambientTemp = 24;
thermalInfo.flux = 0;

simDuration = 1.0;
%% Set default laser settings
w0 = 0.0168; % beam Waist [cm]
focalPoint = 35; % distance from waist to target [cm]
w = @(z) w0 * sqrt(1 + (z.*10.6e-4./(pi*w0.^2)).^2);
I = @(x,y,z,MUA) 2./(w(focalPoint + z).^2.*pi) .* exp(-2.*(x.^2 + y.^2)./(w(focalPoint + z).^2) - MUA.*z);

%% Create two different meshes
% MESH 1
z = [0:0.0015:0.05 0.1:0.05:1];
y = -1:0.025:1;
x = -1:0.025:1;
[X,Y,Z] = meshgrid(x,y,z);
X = X(:); Y = Y(:); Z = Z(:);
nodesPerAxis = [length(x),length(y),length(z)];
boundaryConditions = [0,0,0,0,0,0]';
mesh1 = Mesh(x,y,z,boundaryConditions);

fluenceRate1 = single(I(X,Y,Z,thermalInfo.MUA));
thermalInfo.temperature = 20*ones(prod(nodesPerAxis),1);
simulator.mesh = mesh1;
thermalInfo.fluenceRate = fluenceRate1;
simulator.thermalInfo = thermalInfo;
[Tpred1,sensorTemps1] = simulator.solve(simDuration);
simulator.plotVolume(2);


%% MESH 2
z = [0:0.0015:0.05 0.1:0.05:1];
y = [-1:0.05:-0.5 -0.475:0.025:0.475 0.5:0.05:1];
x = [-1:0.05:-0.5 -0.475:0.025:0.475 0.5:0.05:1];
% x = [-1, -0.8, -0.6, -0.4, -0.2, -0.1 , 0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
% y = [-1, -0.8, -0.6, -0.4, -0.2, -0.1 , 0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
[X,Y,Z] = meshgrid(x,y,z);
X = X(:); Y = Y(:); Z = Z(:);
nodesPerAxis = [length(x),length(y),length(z)];
boundaryConditions = [0,0,0,0,0,0]';
mesh2 = Mesh(x,y,z,boundaryConditions);

fluenceRate2 = single(I(X,Y,Z,thermalInfo.MUA));
thermalInfo.temperature = 20*ones(prod(nodesPerAxis),1);
simulator.mesh = mesh2;
thermalInfo.fluenceRate = fluenceRate2;
simulator.thermalInfo = thermalInfo;
[Tpred2,sensorTemps2] = simulator.solve(simDuration);
simulator.plotVolume(3);


clear mex