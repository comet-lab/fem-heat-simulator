clear; close all; clc
%%
sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.1; 0,0,0.2; 0 0 0.3;0 0 0.4;0 0 0.5];
z = [0:0.0015:0.05 0.1:0.05:1];
x = linspace(-2.5,2.5,101);
y = linspace(-2.5,2.5,101);
nodesPerAxis = [length(x),length(y),length(z)];
boundaryConditions = [1,2,1,1,1,1]';

mesh = Mesh(x,y,z,boundaryConditions);
% Initialization of parameters
simulator = HeatSimulator();
simulator.dt = 0.05;
simulator.alpha = 0.5;
simulator.useAllCPUs = true;
simulator.useGPU = false;
simulator.silentMode = false;
simulator.mesh = mesh;
simulator.sensorLocations = sensorPositions;

thermalInfo = ThermalModel();
thermalInfo.MUA = 400;
thermalInfo.TC = 0.0062;
thermalInfo.VHC = 4.3;
thermalInfo.HTC = 0.008;
thermalInfo.ambientTemp = 24;
thermalInfo.flux = 0;
thermalInfo.temperature = 20*ones(prod(nodesPerAxis),1);
simulator.thermalInfo = thermalInfo;

% set sim duration
timePoints = (0:simulator.dt:10.0)';
% set laser settings
laserPower = ones(size(timePoints));
laserPower(timePoints>5) = 0;
laserPose = zeros(size(timePoints,1),6);
laserPose(:,3) = -35;
w0 = 0.0168;

lambda = 10.6e-4;
laser = Laser(w0,lambda,thermalInfo.MUA);
laser.focalPose = struct('x',0,'y',0,'z',-35,'theta',0,'phi',0,'psi',0);
laser = laser.calculateIrradiance(mesh);
simulator.laser = laser;
%% Simulating 
tic
if ~simulator.silentMode
    fprintf("\n");
end
simulator.buildMatrices = true;
simulator.resetIntegration = true;
% solve can be called with laserPose and laserPower, or without as long as
% calculateIrradiance has been called on the laser
[Tpred,sensorData] = simulator.solve(timePoints,laserPose,laserPower);
toc
%% Plot Sensor Temps over time
simulator.createSensorTempsFigure();

%% Plot volume temperature information
simulator.createVolumetricFigure();
%% plot depth irradiance at final time step
focalPoint = 35; % distance from waist to target [cm]
w = @(z) w0 * sqrt(1 + (z.*lambda./(pi*w0.^2)).^2);
I = @(x,y,z,MUA) 2./(w(focalPoint + z).^2.*pi) .* exp(-2.*(x.^2 + y.^2)./(w(focalPoint + z).^2) - MUA.*z);

[X,Y,Z] = meshgrid(x,y,z);
X = X(:); Y = Y(:); Z = Z(:);
fluenceRate = single(I(X,Y,Z,thermalInfo.MUA));

figure(3);
clf;
tiledlayout('flow');
nexttile()
hold on
plot(z,reshape(I(0,0,z,thermalInfo.MUA),size(z)),...
    'LineWidth',2)
hold off
grid on;
xlabel("Penetration depth (cm)");
ylabel("Normalized Fluence Rate");
title("Normalized Fluence Rate");

%% Plot Temperature Depth at final time step
figure(4);
clf;
hold on
plot(z,Tpred(X==0 & Y==0),...
    'LineWidth',2)
hold off
grid on;
xlabel("Depth (cm)");
ylabel("Temperautre (deg C)");
title("Temperature Prediction with 2 layers");