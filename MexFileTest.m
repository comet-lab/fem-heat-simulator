clear MEX_Heat_Simulation
tissueSize = [1.0,1.0,0.2];
nodeSize = [35,35,101];
ambientTemp = 20;
T0 = single(ambientTemp*ones(nodeSize));
deltaT = 0.05;
tFinal = single(1.0);
w0 = 0.021;
focalPoint = 25;
MUA = 800;
TC = 0.006;
VHC = 4.5;
HTC = 0.075;
useAllCPUs = true;
silentMode = false;
Nn1d = 2;
layer = [tissueSize(3),nodeSize(3)];
sensorPositions = [0,0,0];

w = @(z) w0 * sqrt(1 + (z.*10.6e-4./(pi*w0.^2)).^2);
I = @(x,y,z) 2./(w(focalPoint + z).^2.*pi) .* exp(-2.*((x).^2 + (y).^2)./(w(focalPoint + z).^2) - MUA.*z);
x = linspace(-tissueSize(1)/2,tissueSize(1)/2,nodeSize(1));
y = linspace(-tissueSize(2)/2,tissueSize(2)/2,nodeSize(2));
z = linspace(0,tissueSize(3),nodeSize(3));
[X,Y,Z] = meshgrid(x,y,z);
NFR = single(I(X,Y,Z));
tissueProperties = [MUA,TC,VHC,HTC]';

BC = int32(2*ones(6,1)); %0: HeatSink, 1: Flux, 2: Convection
Jn = 0;
%%
[Tpred,sensorTemps] = MEX_Heat_Simulation(T0,NFR,tissueSize',tFinal,...
    deltaT,tissueProperties,BC,Jn,ambientTemp,sensorPositions,useAllCPUs,...
    silentMode);
sensorTemp = sensorTemps(:,end);
fullSensor = sensorTemps;

%%
figure(1);
clf;
plot(0:deltaT:tFinal,sensorTemps);
grid on;
xlabel("Time (s)");
ylabel("Temperature (s)");