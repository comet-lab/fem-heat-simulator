clc; clear; close all;

clear MEX_Heat_Simulation
%% Initialization of parameters
tissueSize = [5.0,5.0,1.0]; % [cm, cm, cm]
nodesPerAxis = [81,81,50];
ambientTemp = 24; % ambient temp
T0 = single(20*ones(nodesPerAxis)); % initial temp [deg C]
deltaT = 0.05; % Time Step for integration % [s]
alpha = 1/2; % implicit percentage of time integration (0 - forward, 1/2 - crank-nicolson, 1 - backward)
tFinal = single(5.0); % final duration of simulation [s]
time = 0:deltaT:tFinal;
w0 = 0.0168; % beam Waist [cm]
focalPoint = 35; % distance from waist to target [cm]
MUA = 200; % absorption coefficient [cm^-1]
TC = 0.0062; % thermal conductivity [W/K cm]
VHC = 4.3; % volumetric heat capacity [J/K cm^3]
HTC = 0.05; % heat transfer coefficient [W/K cm^2]
useAllCPUs = true; % multithreading enabled
useGPU = true;
silentMode = false; % print statements off
Nn1d = 2; % nodes per axis in a single element
layerInfo = [0.05,30]; % layer height, layer elements
sensorPositions = [0,0,0; 0,0,0.05; -1,0,0; -1/2,0,0];
BC = int32([2,0,0,0,0,0]'); %0: HeatSink, 1: Flux, 2: Convection
flux = 0;
tissueProperties = [MUA,TC,VHC,HTC]';

createMatrices = false; % whether to always recreate all the matrices.

x = linspace(-tissueSize(1)/2,tissueSize(1)/2,nodesPerAxis(1));
y = linspace(-tissueSize(2)/2,tissueSize(2)/2,nodesPerAxis(2));
z = [linspace(0,layerInfo(1)-layerInfo(1)/layerInfo(2),layerInfo(2)) linspace(layerInfo(1),tissueSize(3),nodesPerAxis(3)-layerInfo(2))];
[Y,X,Z] = meshgrid(y,x,z); % left handed coordinate system

% Set time varying laser power and pose
laserPower = ones(1,length(time))*1;
laserPose = [0;0;-35;0;0;0].*ones(6,length(time));
% Comment out lines below to make a constant input
laserPower(time>(tFinal/2)) = 0;
laserPose(1,:) = linspace(-1,1,length(time));
%% Running MutliStep MEX File

tic
if ~silentMode
    fprintf("\n");
end
[TpredMulti,sensorTempsMulti] = MEX_Heat_Simulation_MultiStep(T0,tissueSize',...
    tissueProperties,BC,flux,ambientTemp,sensorPositions,w0,time,...
    laserPose,laserPower,layerInfo,useAllCPUs,useGPU,alpha,...
    silentMode,Nn1d);
fprintf("Running with multistep call: %0.2f sec\n",toc);

%% Running Single Step but calling multiple times to change fluence rate.
clear MEX_Heat_Simulation_MultiStep
pause(5);
silentMode = true;
Tpred = T0;
sensorTemps = zeros(size(sensorTempsMulti));
sensorTemps(:,1) = sensorTempsMulti(:,1);
tic
for i = 2:length(time)
    fluenceRate = calcIrradiance(w0,laserPose(:,i),laserPower(:,i),MUA,X,Y,Z);
    if i == 2
        createMatrices = true; % first call is true to make sure matrices are set up appropriately
    else
        createMatrices = false; % remaining calls can be false or true
        % if true is set and you're using crank-nicolson (alpha = 1/2) then
        % the results won't actually match because creating the matrices
        % resets the explicit step in the crank-nicolson
    end
    [Tpred,sensOutput] = MEX_Heat_Simulation(T0,fluenceRate,tissueSize',deltaT,...
        deltaT,tissueProperties,BC,flux,ambientTemp,sensorPositions,layerInfo,...
        useAllCPUs,useGPU,alpha,silentMode,Nn1d,createMatrices);
    
    sensorTemps(:,i) = sensOutput(:,end);
end
fprintf("Repeated Single Step Calls: %0.2f sec\n",toc);
%% Plot Sensor Temps over time
figure(1);
clf;
hold on;
for ss = 1:size(sensorPositions,1)
    plot(time,sensorTemps(ss,:),'--','LineWidth',2,'DisplayName',...
        sprintf("Single (%g,%g,%g)",sensorPositions(ss,:)));
    plot(time,sensorTempsMulti(ss,:),'-*','LineWidth',1,'DisplayName',...
        sprintf("Multi (%g,%g,%g)",sensorPositions(ss,:)),...
        'MarkerIndices',1:10:length(time));
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
surf(Y(:,:,1),X(:,:,1),TpredMulti(:,:,1))
xlabel("Y Axis (cm)")
ylabel("X Axis (cm)");
title("Suraface Temperature Plot Two-layer mesh")
c = colorbar();
c.Label.String = 'Temperature (deg C)';
axis equal
view(0,90);
colormap('hot')

clear mex

%% Helper Function

function fluenceRate = calcIrradiance(w0,focalPoint,power,MUA,X,Y,Z)

W = w0 * sqrt(1 + ((focalPoint(3) + Z).*10.6e-4./(pi*w0.^2)).^2);
fluenceRate = 2.*power./(W.^2.*pi) .* exp(-2.*((X-focalPoint(1)).^2 + (Y-focalPoint(2)).^2)./(W.^2) - MUA.*Z);
fluenceRate = single(fluenceRate);

end