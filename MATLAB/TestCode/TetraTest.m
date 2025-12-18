clc; clear; close all;
%% 1. Create FEM model for transient heat analysis
fem = femodel('AnalysisType','thermalTransient');

%% 2. Define 3D geometry
fem.Geometry = multicuboid(5, 5, 1); % [cm,cm,cm]
fem.Geometry = addVertex(fem.Geometry,"Coordinates",[0,0,0]);
% fem.Geometry = addVertex(fem.Geometry,"Coordinates",[0,0]);

%% 3. Visualize faces to know IDs
figure(1);
clf
pdegplot(fem.Geometry,'FaceLabels','on','FaceAlpha',0.5);
title('Cuboid Geometry - Face Labels');
axis equal; view(30,30);

%% 7. Generate mesh
fem = generateMesh(fem,'Hmax',0.3,'Hvertex',{9,0.0005},'Hgrad',1.01);
mesh = Mesh(fem,[1,1,1,1,1,1]);
figure(2)
pdemesh(fem,'FaceAlpha',1.0);
%%
sensorPositions = [0,0,0.05];
simulator = HeatSimulator();

simulator.dt = 0.05;
simulator.alpha = 0.5;
simulator.useAllCPUs = true;
simulator.useGPU = false;
simulator.silentMode = false;
simulator.mesh = mesh;
simulator.sensorLocations = sensorPositions;

thermalInfo = ThermalModel();
thermalInfo.MUA = 200;
thermalInfo.TC = 0.0062;
thermalInfo.VHC = 4.3;
thermalInfo.HTC = 0.008;
thermalInfo.ambientTemp = 24;
thermalInfo.flux = 0;
thermalInfo.temperature = 20*ones(size(mesh.nodes,2),1);
simulator.thermalInfo = thermalInfo;

% set sim duration
timePoints = (0:simulator.dt:10.0)';
% set laser settings
w0 = 0.0168;
lambda = 10.6e-4;
laser = Laser(w0,lambda,thermalInfo.MUA);
laser.power = 1;
laser.focalPose = struct('x',0,'y',0,'z',-35,'theta',0,'phi',0,'psi',0);
laser = laser.calculateIrradiance(mesh);
simulator.laser = laser;

%% Our Solver

[Tpred,sensorData] = simulator.solve(timePoints);

%%
simulator.createVolumetricFigure(3);
