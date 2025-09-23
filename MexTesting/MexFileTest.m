clear; close all;
%% Initialization of parameters
useAllCPUs = true; % multithreading enabled
useGPU = false; % enable gpu use
silentMode = false; % print statements off

deltaT = 0.05; % Time Step for integration % [s]
simDuration = single(1.0); % final duration of simulation [s]

tissueSize = [5.0,5.0,1.0]; % [cm, cm, cm]
nodesPerAxis = [101,101,100];
MUA = 200; % absorption coefficient [cm^-1]
TC = 0.0062; % thermal conductivity [W/K cm]
VHC = 4.3; % volumetric heat capacity [J/K cm^3]
HTC = 0.05; % heat transfer coefficient [W/K cm^2]
ambientTemp = 24; % ambient temp
T0 = single(20*ones(nodesPerAxis)); % initial temp [deg C]
alpha = 0.5; % implicit percentage of time integration (0 - forward, 1/2 - crank-nicolson, 1 - backward)
w0 = 0.0168; % beam Waist [cm]
focalPoint = 35; % distance from waist to target [cm]
Nn1d = 2; % nodes per axis in a single element
layerInfo = [0.05,30]; % layer height, layer elements
sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.5; 0,0,0.95; 0 0 1];
BC = int32([2,0,2,2,2,2]'); %0: HeatSink, 1: Flux, 2: Convection
flux = 0;


w = @(z) w0 * sqrt(1 + (z.*10.6e-4./(pi*w0.^2)).^2);
I = @(x,y,z,MUA) 2./(w(focalPoint + z).^2.*pi) .* exp(-2.*(x.^2 + y.^2)./(w(focalPoint + z).^2) - MUA.*z);

x = linspace(-tissueSize(1)/2,tissueSize(1)/2,nodesPerAxis(1));
y = linspace(-tissueSize(2)/2,tissueSize(2)/2,nodesPerAxis(2));
z = [linspace(0,layerInfo(1)-layerInfo(1)/layerInfo(2),layerInfo(2)) linspace(layerInfo(1),tissueSize(3),nodesPerAxis(3)-layerInfo(2))];
[X,Y,Z] = meshgrid(x,y,z);
fluenceRate = single(I(X,Y,Z,MUA));
tissueProperties = [MUA,TC,VHC,HTC]';

createMatrices = true; % whether to always recreate all the matrices. 
%% Running MEX File

tic
if ~silentMode
    fprintf("\n");
end
[Tpred,sensorTemps] = MEX_Heat_Simulation(T0,fluenceRate,tissueSize',simDuration,...
    deltaT,tissueProperties,BC,flux,ambientTemp,sensorPositions,layerInfo,...
    useAllCPUs,useGPU,alpha,silentMode,Nn1d,createMatrices);
toc
clear mex
%% Plot Sensor Temps over time
figure(1);
clf;
hold on;
for ss = 1:size(sensorPositions,1)
    plot(0:deltaT:simDuration,sensorTemps(ss,:),'LineWidth',2,'DisplayName',...
        sprintf("(%g,%g,%g)",sensorPositions(ss,:)));
end
hold off
grid on;
xlabel("Time (s)");
ylabel("Temperature (deg C)");
title("Sensor temperature over time");
legend()
%% plot depth irradiance at final time step
figure(2);
clf;
tiledlayout('flow');
nexttile()
hold on
plot(z,reshape(fluenceRate(floor(nodesPerAxis(1)/2),floor(nodesPerAxis(2)/2),:),size(z)),...
    'LineWidth',2,'DisplayName',"Two Layers")
hold off
grid on;
legend()
xlabel("Penetration depth (cm)");
ylabel("Normalized Fluence Rate");
title("Normalized Fluence Rate with 2 Layers");
nexttile()
hold on
plot(z(1:30),reshape(fluenceRate(floor(nodesPerAxis(1)/2),floor(nodesPerAxis(2)/2),1:30),size(z(1:30))),...
    'LineWidth',2,'DisplayName',"Two Layers")
hold off
grid on;
legend()
xlabel("Penetration depth (cm)");
ylabel("Normalized Fluence Rate");
title("Normalized Fluence Rate with 2 Layers");

%% Plot Temperature Depth at final time step
figure(3);
clf;
hold on
plot(z,reshape(Tpred(floor(nodesPerAxis(1)/2),floor(nodesPerAxis(2)/2),:),size(z)),...
    'LineWidth',2,'DisplayName',sprintf("Two Layers"))
hold off
grid on;
xlabel("Depth (cm)");
ylabel("Temperautre (deg C)");
title("Temperature Prediction with 2 layers");
legend()
%% Surface Temperature Plot
figure(4)
clf;
tiledlayout('flow')
nexttile()
surf(X(:,:,1),Y(:,:,1),Tpred(:,:,1))
xlabel("X Axis (cm)")
ylabel("Y Axis (cm)");
title("Suraface Temperature Plot Two-layer mesh")
c = colorbar();
c.Label.String = 'Temperature (deg C)';
axis equal
view(0,90);
colormap('hot')