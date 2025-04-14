clear MEX_Heat_Simulation
%%
tissueSize = [2.0,2.0,1.0];
nodeSize = [41,41,50];
ambientTemp = 24;
T0 = single(20*ones(nodeSize));
deltaT = 0.05;
tFinal = single(1.0);
w0 = 0.0168;
focalPoint = 35;
MUA = 25;
TC = 0.0062;
VHC = 4.3;
HTC = 0.05;
useAllCPUs = false;
silentMode = true;
Nn1d = 2;
layerInfo = [0.05,30];
sensorPositions = [0,0,0; 0 0 0.05; 0 0 0.5; 0,0,0.95; 0 0 1];

w = @(z) w0 * sqrt(1 + (z.*10.6e-4./(pi*w0.^2)).^2);
I = @(x,y,z) 2./(w(focalPoint + z).^2.*pi) .* exp(-2.*(x.^2 + y.^2)./(w(focalPoint + z).^2) - MUA.*z);

xLayer = linspace(-tissueSize(1)/2,tissueSize(1)/2,nodeSize(1));
yLayer = linspace(-tissueSize(2)/2,tissueSize(2)/2,nodeSize(2));
zLayer = [linspace(0,layerInfo(1)-layerInfo(1)/layerInfo(2),layerInfo(2)) linspace(layerInfo(1),tissueSize(3),nodeSize(3)-layerInfo(2))];
[X,Y,Z] = meshgrid(xLayer,yLayer,zLayer);
NFRLayer = single(I(X,Y,Z));
tissueProperties = [MUA,TC,VHC,HTC]';

BC = int32([2,0,0,0,0,0]'); %0: HeatSink, 1: Flux, 2: Convection
Jn = 0;
tic
for j = 1:2
parfor i = 1:6
fprintf("\n\n");

tempTP = tissueProperties
tempTP(1) = i*10;
[~,~] = MEX_Heat_Simulation(T0,NFRLayer,tissueSize',tFinal,...
    deltaT,tempTP,BC,Jn,ambientTemp,sensorPositions,useAllCPUs,...
    silentMode,layerInfo);

end
end
toc

figure(1);
clf;
hold on;
for ss = 1:size(sensorPositions,1)
    plot(0:deltaT:tFinal,sensorTempsLayer(ss,:),'LineWidth',2,'DisplayName',...
        sprintf("(%g,%g,%g)",sensorPositions(ss,:)));
end
hold off
grid on;
xlabel("Time (s)");
ylabel("Temperature (deg C)");
title("Sensor temperature over time");
legend()
%%
figure(2);
clf;
tiledlayout('flow');
nexttile()
hold on
plot(zLayer,reshape(NFRLayer(17,17,:),size(zLayer)),'LineWidth',2,'DisplayName',"Two Layers")
% plot(z,reshape(NFR(17,17,:),size(z)),'LineWidth',2,'DisplayName',"One Layers")
hold off
grid on;
legend()
xlabel("Penetration depth (cm)");
ylabel("Normalized Fluence Rate");
title("Normalized Fluence Rate with 2 Layers");
nexttile()
hold on
plot(zLayer(1:30),reshape(NFRLayer(17,17,1:30),size(zLayer(1:30))),'LineWidth',2,'DisplayName',"Two Layers")
% plot(z(1:30),reshape(NFR(17,17,1:30),size(z(1:30))),'LineWidth',2,'DisplayName',"One Layers")
hold off
grid on;
legend()
xlabel("Penetration depth (cm)");
ylabel("Normalized Fluence Rate");
title("Normalized Fluence Rate with 2 Layers");

figure(3);
clf;
hold on
plot(zLayer,reshape(TpredLayer(17,17,:),size(zLayer)),'LineWidth',2,'DisplayName',sprintf("Two Layers"))
% plot(z,reshape(Tpred(17,17,:),size(z)),'LineWidth',2,'DisplayName',sprintf("One Layer"))
hold off
grid on;
xlabel("Depth (cm)");
ylabel("Temperautre (deg C)");
title("Temperature Prediction with 2 layers");
legend()
%%
figure(4)
clf;
tiledlayout('flow')
nexttile()
surf(X(:,:,1),Y(:,:,1),TpredLayer(:,:,1))
xlabel("X Axis (cm)")
ylabel("Y Axis (cm)");
title("Suraface Temperature Plot Two-layer mesh")
c = colorbar();
c.Label.String = 'Temperature (deg C)';
axis equal
view(0,90);
colormap('hot')


nexttile()
surf(X(:,:,1),Y(:,:,1),Tpred(:,:,1))
xlabel("X Axis (cm)")
ylabel("Y Axis (cm)");
title("Suraface Temperature Plot One-layer mesh")
c = colorbar();
c.Label.String = 'Temperature (deg C)';
axis equal
view(0,90);
colormap('hot')
% h_f2 = plotVolumetric.plotVolumetric(2,x,y,z,Tpred,'MCmatlab_fromZero');
% title(sprintf('Cpp Final Temp, [C]'))