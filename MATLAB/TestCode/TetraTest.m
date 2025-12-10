%% 1. Create FEM model for transient heat analysis
fem = femodel('AnalysisType','thermalTransient');

%% 2. Define 3D geometry
gm = multicuboid(5, 5, 1); % [cm,cm,cm]
fem.Geometry = gm; % 
%% 3. Visualize faces to know IDs
figure(1);
clf
pdegplot(fem.Geometry,'FaceLabels','on','FaceAlpha',0.5);
title('Cuboid Geometry - Face Labels');
axis equal; view(30,30);

%% 4. Define material properties
% fem.MaterialProperties = materialProperties('ThermalConductivity',0.62, ... % W/m C : 0.0062 W/cm C -> 
%     'MassDensity', 1000,... % kg/m^3 : 1 g/cm^3 -> 0.001 kg/cm^3 -> 1000 kg/m^3 
%     'SpecificHeat',4000); % J/kg C : 4.0 J/g C -> 4000 J/kg C

%% 5. Dirichlet Boundaries
% Face 1: 100°C
% fem.FaceBC(1) = faceBC("Temperature",20);

%% Heat flux Boundary
% fem.FaceLoad(2:6) = faceLoad("ConvectionCoefficient",100,... % W/m^2 C : 0.01 W/cm^2 C -> 100 W/m^2 C
%     "AmbientTemperature",25);

%% 6. Initial condition
% fem.VertexIC = vertexIC("Temperature",25);
% fem.CellIC = cellIC("Temperature",25);

%% 7. Generate mesh
fem = generateMesh(fem,'Hmax',0.2);
mesh = Mesh(fem,[1,1,1,1,1,1]);
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
thermalInfo.MUA = 2;
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
