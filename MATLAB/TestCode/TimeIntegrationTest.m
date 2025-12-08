clc; clear; close all;

clear MEX_Heat_Simulation
%% Initialization of parameters
sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.1; 0,0,0.2];
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
simulator.silentMode = true;
simulator.buildMatrices = true;
simulator.resetIntegration = true;
simulator.mesh = mesh;
simulator.sensorLocations = sensorPositions;

thermalInfo = ThermalModel();
thermalInfo.MUA = 400;
thermalInfo.TC = 0.0062;
thermalInfo.VHC = 4.3;
thermalInfo.HTC = 0.008;
thermalInfo.ambientTemp = 24;
thermalInfo.flux = 0;
T0 = 20*ones(prod(nodesPerAxis),1);
thermalInfo.temperature = T0;
simulator.thermalInfo = thermalInfo;

w0 = 0.0168;
lambda = 10.6e-4;
laser = Laser(w0,lambda,thermalInfo.MUA);
laser.focalPose = struct('x',0,'y',0,'z',-35,'theta',0,'phi',0,'psi',0);
laser = laser.calculateIrradiance(mesh);
simulator.laser = laser;
%% Running MEX File
tFinal = 1;
deltaTOpts = [0.05, 0.2, 0.5, 1.0]; % Time Step for integration % [s]
alphaOpts = [1/2, 1]; % implicit percentage of time integration (0 - forward, 1/2 - crank-nicolson, 1 - backward)
tic
sensorTempsCell = cell(length(deltaTOpts),length(alphaOpts));
durationCell = cell(length(deltaTOpts),length(alphaOpts));
for dd = 1:length(deltaTOpts)
    deltaT = deltaTOpts(dd);
    for aa = 1:length(alphaOpts)
        alpha = alphaOpts(aa);
        if ~simulator.silentMode
            fprintf("\n");
        end
        tic
        timePoints = (0:deltaT:tFinal)';
        simulator.dt = deltaT;
        simulator.alpha = alpha;
        simulator.thermalInfo.temperature = T0;
        [Tpred,sensorTemps] = simulator.solve(timePoints);
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


            plot(0:deltaT:tFinal,sensorTempsCell{dd,aa}(:,ss),'LineStyle',lineOpts{dd},...
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