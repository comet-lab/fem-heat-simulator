clc; clear; close all;

clear MEX_Heat_Simulation
%% Initialization of parameters
tissueSize = [5.0,5.0,1.0];
nodesPerAxis = [81,81,50];
ambientTemp = 24;
T0 = single(20*ones(nodesPerAxis));
deltaT = 0.05;
tFinal = single(0.05);
w0 = 0.0168;
focalPoint = 35;
MUA = 200;
TC = 0.0062;
VHC = 4.3;
HTC = 0.05;
useAllCPUs = true;
silentMode = false;
Nn1d = 2;
layerInfo = [0.05,30];
sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.5; 0,0,0.95; 0 0 1];

w = @(z) w0 * sqrt(1 + (z.*10.6e-4./(pi*w0.^2)).^2);
I = @(x,y,z,MUA) 2./(w(focalPoint + z).^2.*pi) .* exp(-2.*(x.^2 + y.^2)./(w(focalPoint + z).^2) - MUA.*z);

x = linspace(-tissueSize(1)/2,tissueSize(1)/2,nodesPerAxis(1));
y = linspace(-tissueSize(2)/2,tissueSize(2)/2,nodesPerAxis(2));
z = [linspace(0,layerInfo(1)-layerInfo(1)/layerInfo(2),layerInfo(2)) linspace(layerInfo(1),tissueSize(3),nodesPerAxis(3)-layerInfo(2))];
[X,Y,Z] = meshgrid(x,y,z);
fluenceRate = single(I(X,Y,Z,MUA));
tissueProperties = [MUA,TC,VHC,HTC]';

BC = int32([2,0,0,0,0,0]'); %0: HeatSink, 1: Flux, 2: Convection
flux = 0;
createMatrices = true;
%% Running MEX File

tic
if ~silentMode
    fprintf("\n");
end
[Tpred,sensorTemps] = MEX_Heat_Simulation(T0,fluenceRate,tissueSize',tFinal,...
    deltaT,tissueProperties,BC,flux,ambientTemp,sensorPositions,useAllCPUs,...
    silentMode,layerInfo,Nn1d,createMatrices);
toc

%% Plot Sensor Temps over time
figure(1);
clf;
hold on;
for ss = 1:size(sensorPositions,1)
    plot(0:deltaT:tFinal,sensorTemps(ss,:),'LineWidth',2,'DisplayName',...
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

clear mex