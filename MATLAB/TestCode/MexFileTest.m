clear; close all; clc

sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.5; 0,0,0.95; 0 0 1];
z = 0:0.02:1;
y = -1:0.025:1;
x = -1:0.025:1;
nodesPerAxis = [length(x),length(y),length(z)];

% Initialization of parameters
simulator = HeatSimulator();
simulator.dt = 0.05;
simulator.alpha = 0.5;
simulator.useAllCPUs = true;
simulator.useGPU = false;
simulator.silentMode = false;
simulator.xpos = x;
simulator.ypos = y;
simulator.zpos = z;
simulator.boundaryConditions = [0,0,0,0,0,0]';
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

[X,Y,Z] = meshgrid(simulator.xpos,simulator.ypos,simulator.zpos);
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
% try
    [Tpred,sensorTemps] = simulator.solve(simDuration);
% catch exception
%     disp(exception.message)
%     for i = 1:length(exception.stack)
%         fprintf("Line %d in %s\n",exception.stack(i).line,exception.stack(i).file);
%     end
% end

toc
clear mex
%% Plot Sensor Temps over time
figure(1);
clf;
hold on;
for ss = 1:size(sensorPositions,1)
    plot(0:deltaT:simDuration,sensorTemps(:,ss),'LineWidth',2,'DisplayName',...
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
plot(z,reshape(I(0,0,z,thermalInfo.MUA),size(z)),...
    'LineWidth',2)
hold off
grid on;
xlabel("Penetration depth (cm)");
ylabel("Normalized Fluence Rate");
title("Normalized Fluence Rate");

%% Plot Temperature Depth at final time step
figure(3);
clf;
hold on
plot(z,Tpred(X==0 & Y==0),...
    'LineWidth',2)
hold off
grid on;
xlabel("Depth (cm)");
ylabel("Temperautre (deg C)");
title("Temperature Prediction with 2 layers");
%% Surface Temperature Plot
figure(4)
clf;
tiledlayout('flow')
nexttile()
Xs = reshape(X(Z == 0),[nodesPerAxis(1),nodesPerAxis(2)]);
Ys = reshape(Y(Z == 0),[nodesPerAxis(1),nodesPerAxis(2)]);
Ts = reshape(Tpred(Z == 0),[nodesPerAxis(1),nodesPerAxis(2)]);
surf(Xs,Ys,Ts);
xlabel("X Axis (cm)")
ylabel("Y Axis (cm)");
title("Suraface Temperature Plot Two-layer mesh")
c = colorbar();
c.Label.String = 'Temperature (deg C)';
axis equal
view(0,90);
colormap('hot')