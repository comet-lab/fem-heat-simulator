clear; close all; clc

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
simulator.useAllCPUs = false;
simulator.useGPU = false;
simulator.silentMode = false;

simulator.sensorLocations = sensorPositions;

thermalInfo = ThermalModel();
thermalInfo.MUA = 400;
thermalInfo.TC = 0.0062;
thermalInfo.VHC = 4.3;
thermalInfo.HTC = 0.008;
thermalInfo.ambientTemp = 24;
thermalInfo.flux = 0;

simDuration = 15.0;
%% Set default laser settings
w0 = 0.0167; % beam Waist [cm]
focalPoint = 35; % distance from waist to target [cm]
w = @(z) w0 * sqrt(1 + (z.*10.6e-4./(pi*w0.^2)).^2);
I = @(x,y,z,MUA) 2./(w(focalPoint + z).^2.*pi) .* exp(-2.*(x.^2 + y.^2)./(w(focalPoint + z).^2) - MUA.*z);

%% Create two different meshes
% MESH 1
z1 = [0:0.0015:0.05 0.1:0.05:1];
y1 = -1:0.025:1;
x1 = -1:0.025:1;
[X1,Y1,Z1] = meshgrid(x1,y1,z1);
nodesPerAxis = [length(x1),length(y1),length(z1)];
boundaryConditions = [1,1,1,1,1,1]';
mesh1 = Mesh(x1,y1,z1,boundaryConditions);

fluenceRate1 = single(I(X1(:),Y1(:),Z1(:),thermalInfo.MUA));
thermalInfo.temperature = 20*ones(prod(nodesPerAxis),1);
simulator.mesh = mesh1;
thermalInfo.fluenceRate = fluenceRate1;
simulator.thermalInfo = thermalInfo;
[Tpred1,sensorTemps1] = simulator.solve(simDuration);


%% Second Simulator
simulator2 = simulator.deepCopy();
z2 = [0:0.0015:0.05 0.1:0.05:1];
y2 = [Mesh.geometricSpacing(-1,-0.3,15,0.075) (-0.275:0.025:0.275) Mesh.geometricSpacing(0.3,1,15,0.075,true)];
x2 = [Mesh.geometricSpacing(-1,-0.3,15,0.075) (-0.275:0.025:0.275) Mesh.geometricSpacing(0.3,1,15,0.075,true)];
[X2,Y2,Z2] = meshgrid(x2,y2,z2);
nodesPerAxis = [length(x2),length(y2),length(z2)];
boundaryConditions = [1,1,1,1,1,1]';
mesh2 = Mesh(x2,y2,z2,boundaryConditions);

simulator2.mesh = mesh2;
%%
fluenceRate2 = single(I(X2(:),Y2(:),Z2(:),thermalInfo.MUA));
simulator2.thermalInfo.temperature = 20*ones(prod(nodesPerAxis),1);
simulator2.thermalInfo.fluenceRate = fluenceRate2;
[Tpred2,sensorTemps2] = simulator2.solve(simDuration);


%% Fluence Comparison

figure(1)
clf;
tiledlayout
nexttile()
hold on
plot(-1:0.001:1,I(-1:0.001:1,0,0,thermalInfo.MUA),'DisplayName','Ground Truth',...
    'LineWidth',2,'LineStyle','-')
plot(x1,I(x1,0,0,thermalInfo.MUA),'DisplayName',"Sim 1 Fluence Rate",...
    'LineWidth',2,'LineStyle','--')
plot(x2,I(x2,0,0,thermalInfo.MUA),'DisplayName',"Sim 2 Fluence Rate",...
    'LineWidth',2,'LineStyle',':')

xlabel("X Axis (cm)")
ylabel("Irradiance (W/cm^2)")
grid on
legend()
%% Volumetric plots
simulator.createVolumetricFigure(3);
simulator2.createVolumetricFigure(4);

%%
HeatSimulator.compareSimSensors(5,simulator,simulator2)

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
