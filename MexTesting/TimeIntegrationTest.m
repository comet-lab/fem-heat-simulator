clc; clear; close all;

clear MEX_Heat_Simulation
%% Initialization of parameters
tissueSize = [5.0,5.0,1.0]; % [cm, cm, cm]
nodesPerAxis = [81,81,50];
ambientTemp = 24; % ambient temp
T0 = single(20*ones(nodesPerAxis)); % initial temp [deg C]
tFinal = single(15.0); % final duration of simulation [s]
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
sensorPositions = [0,0,0; 0 0.5 0; 0 0 0.5];
BC = int32([2,0,0,0,0,0]'); %0: HeatSink, 1: Flux, 2: Convection
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
deltaTOpts = [0.05, 0.2, 0.5, 1.0]; % Time Step for integration % [s]
alphaOpts = [1/2, 1]; % implicit percentage of time integration (0 - forward, 1/2 - crank-nicolson, 1 - backward)
tic
sensorTempsCell = cell(length(deltaTOpts),length(alphaOpts));
durationCell = cell(length(deltaTOpts),length(alphaOpts));
for dd = 1:length(deltaTOpts)
    deltaT = deltaTOpts(dd);
    for aa = 1:length(alphaOpts)
        alpha = alphaOpts(aa);
        if ~silentMode
            fprintf("\n");
        end
        tic
        [Tpred,sensorTemps] = MEX_Heat_Simulation(T0,fluenceRate,tissueSize',tFinal,...
            deltaT,tissueProperties,BC,flux,ambientTemp,sensorPositions,layerInfo,...
            useAllCPUs,useGPU,alpha,silentMode,Nn1d,createMatrices);
        durationCell{dd,aa} = toc;

        sensorTempsCell{dd,aa} = sensorTemps;
    end
end

%% Plot Sensor Temps over time
fprintf("\n\n")
c = colororder;
lineOpts = {'-', '--',':','-.'};
figure(1);
clf;
tl = tiledlayout('flow');
title(tl,'Comparison of Time-integration and step size');
for ss = 1:size(sensorPositions,1)
    nexttile()
    hold on;
    for aa = 1:length(alphaOpts)

        alpha = alphaOpts(aa);
        for dd = 1:length(deltaTOpts)
            deltaT = deltaTOpts(dd);


            plot(0:deltaT:tFinal,sensorTempsCell{dd,aa}(ss,:),'LineStyle',lineOpts{dd},...
                'LineWidth',1.5,'Color',c(aa,:),'DisplayName',...
                sprintf("\\alpha: %g, \\Deltat: %g",alpha,deltaT));

            if ss == 1
                fprintf("Duration for alpha = %g and deltaT= %g: %0.1f s\n",alpha,deltaT,durationCell{dd,aa});
            end
        end
    end
    set(gca,'FontSize',14)
    hold off
    title(sprintf("(%g,%g,%g)",sensorPositions(ss,:)))
    xlabel("Time (s)");
    ylabel("Temperature (deg C)");
    grid on;
end
leg = legend('Orientation','horizontal');
leg.Layout.Tile = 'south';


clear mex