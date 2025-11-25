clear; close all; clc
%%
sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.5; 0,0,0.95; 0 0 1];
z = 0:0.02:1;
y = -1:0.025:1;
x = -1:0.025:1;
nodesPerAxis = [length(x),length(y),length(z)];
boundaryConditions = [0,0,0,0,0,0]';

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
thermalInfo.MUA = 200;
thermalInfo.TC = 0.0062;
thermalInfo.VHC = 4.3;
thermalInfo.HTC = 0.01;
thermalInfo.ambientTemp = 24;
thermalInfo.flux = 0;
thermalInfo.temperature = 20*ones(prod(nodesPerAxis),1);

w0 = 0.0168; % beam Waist [cm]
focalPoint = 35; % distance from waist to target [cm]
w = @(z) w0 * sqrt(1 + (z.*10.6e-4./(pi*w0.^2)).^2);
I = @(x,y,z,MUA) 2./(w(focalPoint + z).^2.*pi) .* exp(-2.*(x.^2 + y.^2)./(w(focalPoint + z).^2) - MUA.*z);

[X,Y,Z] = meshgrid(x,y,z);
X = X(:); Y = Y(:); Z = Z(:);
fluenceRate = single(I(X,Y,Z,thermalInfo.MUA));
thermalInfo.fluenceRate = fluenceRate;


simulator.thermalInfo = thermalInfo;


% Running MEX File

tic
simDuration = 1.0;
deltaT = simulator.dt;

if ~simulator.silentMode
    fprintf("\n");
end
simulator.buildMatrices = false;
simulator.resetIntegration = true;
[Tpred,sensorTemps] = simulator.solve(simDuration);

toc

%% Plot Sensor Temps over time
simulator.plotSensorTemps(sensorTemps);

%% Plot volume temperature information
simulator.plotVolume();
%% plot depth irradiance at final time step
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