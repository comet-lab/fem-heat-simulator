clear; close all; clc
simDuration = 15.0;
tissueSize = [5,5,1];
sensorPositions = [ 0,   0,  0;
                    0.1, 0   0;
                    0.2, 0   0;
                    0.3, 0,  0;
                    0,   0   0.05;
                    0,   0   0.1;
                    0,   0   0.2;
                    0,   0   0.5;];
depthSensorIdx = 4; % index where we switch from radial to depth sensor

% Initialization of parameters
simulator = HeatSimulator();
simulator.dt = 0.05;
simulator.alpha = 0.5;
simulator.useAllCPUs = true;
simulator.useGPU = false;
simulator.silentMode = true;

simulator.sensorLocations = sensorPositions;

thermalInfo = ThermalModel();
thermalInfo.MUA = 400;
thermalInfo.TC = 0.0062;
thermalInfo.VHC = 4.3;
thermalInfo.HTC = 0.008;
thermalInfo.ambientTemp = 24;
thermalInfo.flux = 0;
simulator.thermalInfo = thermalInfo;

% Set default laser settings
% set laser settings
% laserPower = ones(size(timePoints));
% laserPower(timePoints>5) = 0;
% laserPose = zeros(size(timePoints,1),6);
% laserPose(:,3) = -35;

w0 = 0.0168;
lambda = 10.6e-4;
focalDist = 35;
laser = Laser(w0,lambda,thermalInfo.MUA);
laser.focalPose = struct('x',0,'y',0,'z',-focalDist,'theta',0,'phi',0,'psi',0);
simulator.laser = laser;

%
timePoints = (0:simulator.dt:simDuration)';

%% Create two different meshes
% MESH 1
z1 = [0:0.0010:0.05 0.1:0.05:tissueSize(3)];
y1 = linspace(-tissueSize(2)/2,tissueSize(2)/2,201);
x1 = linspace(-tissueSize(1)/2,tissueSize(1)/2,201);
[X1,Y1,Z1] = meshgrid(x1,y1,z1);
nodesPerAxis1 = [length(x1),length(y1),length(z1)];
boundaryConditions = [1,1,1,1,1,1]';
mesh1 = Mesh(x1,y1,z1,boundaryConditions);

simulator.thermalInfo.temperature = 20*ones(prod(nodesPerAxis1),1);
simulator.laser = simulator.laser.calculateIrradiance(mesh1);
simulator.mesh = mesh1;

tic
[Tpred1,sensorTemps1] = simulator.solve(timePoints);
fprintf("Mesh 1 duration: %0.2f sec\n", toc);

%% Second Simulator
simulator2 = simulator.deepCopy();
nodesPerAxis2 = [51,51,35];
hMax = [0.3,0.3,0.15];
hMin = [0.025,0.025,0.0005];
[x2,y2,z2] = Mesh.getNodesGeometric(tissueSize,nodesPerAxis2,hMax,hMin);
[X2,Y2,Z2] = meshgrid(x2,y2,z2);
nodesPerAxis2 = [length(x2),length(y2),length(z2)];
boundaryConditions = [1,1,1,1,1,1]';
mesh2 = Mesh(x2,y2,z2,boundaryConditions);

simulator2.mesh = mesh2;
simulator2.thermalInfo.temperature = 20*ones(prod(nodesPerAxis2),1);
simulator2.laser = simulator2.laser.calculateIrradiance(mesh2);

tic
[Tpred2,sensorTemps2] = simulator2.solve(timePoints);
fprintf("Mesh 2 duration: %0.2f sec\n", toc);

%% Third Simulator
simulator3 = simulator.deepCopy();
nodesPerAxis3 = [41,41,30];
hMax = [0.3,0.3,0.2];
hMin = [0.025,0.025,0.0005];
[x3,y3,z3] = Mesh.getNodesGeometric(tissueSize,nodesPerAxis3,hMax,hMin);
[X3,Y3,Z3] = meshgrid(x3,y3,z3);
boundaryConditions = [1,1,1,1,1,1]';
mesh3 = Mesh(x3,y3,z3,boundaryConditions);

simulator3.mesh = mesh3;
simulator3.thermalInfo.temperature = 20*ones(prod(nodesPerAxis3),1);
simulator3.laser = simulator3.laser.calculateIrradiance(mesh3);

tic
[Tpred3,sensorTemps3] = simulator3.solve(timePoints);
fprintf("Mesh 2 duration: %0.2f sec\n", toc);

%% Fluence Comparison
w = @(z) w0.*sqrt( 1 + (lambda.*(z+focalDist)./(pi*w0.^2)).^2);
I = @(x,y,z,MUA) 2./(pi*w(z).^2).*exp( -2 .* (x.^2 + y.^2)./w(z).^2 - MUA.*z);
figure(1)
clf;
tiledlayout(1,2)
nexttile()
hold on
plot(-1:0.001:1,I(-1:0.001:1,0,0,thermalInfo.MUA),'DisplayName','Ground Truth',...
    'LineWidth',2,'LineStyle','-')
plot(x1,I(x1,0,0,thermalInfo.MUA),'DisplayName',"Sim 1 Fluence Rate",...
    'LineWidth',2,'LineStyle','--')
plot(x2,I(x2,0,0,thermalInfo.MUA),'DisplayName',"Sim 2 Fluence Rate",...
    'LineWidth',2,'LineStyle',':')
plot(x3,I(x3,0,0,thermalInfo.MUA),'DisplayName',"Sim 3 Fluence Rate",...
    'LineWidth',2,'LineStyle','-.')

xlabel("X Axis (cm)")
ylabel("Irradiance (W/cm^2)")
grid on
legend()

nexttile()
hold on;
plot(0:0.0001:1,I(0,0,0:0.0001:1,thermalInfo.MUA),'DisplayName','Ground Truth',...
    'LineWidth',2,'LineStyle','-')
plot(z1,I(0,0,z1,thermalInfo.MUA),'DisplayName',"Sim 1 Fluence Rate",...
    'LineWidth',2,'LineStyle','--')
plot(z2,I(0,0,z2,thermalInfo.MUA),'DisplayName',"Sim 2 Fluence Rate",...
    'LineWidth',2,'LineStyle',':')
plot(z3,I(0,0,z3,thermalInfo.MUA),'DisplayName',"Sim 3 Fluence Rate",...
    'LineWidth',2,'LineStyle',':')
xlabel("Z Axis (cm)")
ylabel("Irradiance (W/cm^2)")
% set(gca,'XScale','log');
grid on;

%% Volumetric plots
% simulator.createVolumetricFigure(3);
% simulator2.createVolumetricFigure(4);

%%
HeatSimulator.compareSimSensors(5,simulator,simulator2,simulator3)

%% plot node density in x-y and z-x

figure(6);
clf
hold on
% mesh 1
% horizontal lines (rows)
plot(X1(:,:,1)', Y1(:,:,1)', 'r','LineWidth',1);
% vertical lines (columns)
plot(X1(:,:,1),  Y1(:,:,1),  'r','LineWidth',1);

% mesh 1
% horizontal lines (rows)
plot(X2(:,:,1)', Y2(:,:,1)', 'k','LineWidth',2);
% vertical lines (columns)
plot(X2(:,:,1),  Y2(:,:,1),  'k','LineWidth',2);

axis equal
grid on
title("X-Y Grid Comparison")
xlabel("X Axis")
ylabel("Y Axis")

figure(7);
clf
hold on
% mesh 1
surf(squeeze(X1(1,:,:)),squeeze(Y1(1,:,:)),squeeze(Z1(1,:,:)),'FaceAlpha',0.5,...
    'LineWidth',1,'FaceColor','b')
% mesh 2
surf(squeeze(X2(1,:,:)),squeeze(Y2(1,:,:)),squeeze(Z2(1,:,:)),'FaceAlpha',0.5,...
    'LineWidth',2,'FaceColor','r')
% axis equal
grid on
title("Z-Y Grid Comparison")
zlabel("z Axis")
ylabel("Y Axis")
xlabel("X Axis")
set(gca,'ZDir','reverse')
view(0,0)
