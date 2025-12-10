clear; close all;  

clear MEX_Heat_Simulation
clear MEX_Heat_Simulation_MultiStep
%% Initialization of parameters
useAllCPUs = true;
useGPU = false;
silentMode = true;
alpha = 0.5;
numTimeSteps = 10;
dt = 0.05;

% Timing Tests
[CaseSensorTemps, durationVec] = runMexTimeTest(useAllCPUs, useGPU, alpha,...
    silentMode=silentMode,numTimeSteps=numTimeSteps,deltaT=dt);
nCases = size(CaseSensorTemps,1);

%% Plot Sensor Temps to confirm the different methods produce the same result
sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.1; 0,0,0.2; 0 0 0.3;0 0 0.4;0 0 0.5];
time = 0:dt:dt*numTimeSteps;
figure(1);
clf;
tl = tiledlayout('flow');
lineStyles = {'-','--','-.',':'};
for ss = 1:size(sensorPositions,1)
    ax = nexttile();
    for cc = 1:nCases
        hold on;
        plot(time,reshape(CaseSensorTemps(cc,ss,:),size(time)),'LineWidth',2,'DisplayName',...
            sprintf("Case %d",cc),'LineStyle',lineStyles{cc});
        hold off
    end
    title(sprintf("Location (%g, %g, %g)",sensorPositions(ss,:)));
    xlabel("Temperature (deg C)");
    ylabel("Time (s)");
    grid on;
end
leg = legend('Orientation','horizontal');
leg.Layout.Tile = 'south';