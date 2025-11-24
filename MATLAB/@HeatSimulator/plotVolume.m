function plotVolume(obj,figNum)
%PLOTVOLUME Plots volumetric temperature data
%TODO: This function needs to be updated to plot arbitrary meshes. 
arguments (Input)
    obj
    figNum (1,1) double = 2
end


if obj.mesh.useXYZ
    [Y,X,Z] = meshgrid(obj.mesh.ypos,obj.mesh.xpos,obj.mesh.zpos);
    Nx = length(obj.mesh.xpos);
    Ny = length(obj.mesh.ypos);
    Nz = length(obj.mesh.zpos);
end
data = reshape(obj.thermalInfo.temperature,[Nx,Ny,Nz]);

figure(figNum)
clf;
hold on
% surf(X(:,:,1),Y(:,:,1),Z(:,:,1),data(:,:,1),...
%     'FaceColor','interp','EdgeColor','none');
% surf(reshape(X(:,1,:),Nx,[]),reshape(Y(:,1,:),Nx,Nz),reshape(Z(:,1,:),Nx,Nz), reshape(data(:,1,:),Nx,Nz),...
%     'EdgeColor','none');
% surf(reshape(X(1,:,:),Ny,Nz),reshape(Y(1,:,:),Ny,Nz),reshape(Z(1,:,:),Ny,Nz),reshape(data(1,:,:),Ny,Nz),...
%     'EdgeColor','none');


% Initial slices
sx = round(Nx/2);
sy = round(Ny/2);
sz = round(1);
hSliceX = surf(squeeze(X(sx,:,:)), squeeze(Y(sx,:,:)),squeeze(Z(sx,:,:)), squeeze(data(sx,:,:)), ...
    'FaceColor','interp','EdgeColor','interp');
hSliceY = surf(squeeze(X(:,sy,:)), squeeze(Y(:,sy,:)), squeeze(Z(:,sy,:)), ...
    squeeze(data(:,sy,:)), 'EdgeColor','interp','FaceColor','interp');
hSliceZ = surf(X(:,:,sz), Y(:,:,sz), Z(:,:,sz), data(:,:,sz), ...
    'EdgeColor','interp','FaceColor','interp');


hold off
xlabel("X Axis (cm)")
ylabel("Y Axis (cm)")
zlabel("Z Axis (cm)")
title("Volumetric Temperature Information");
cb = colorbar;
colormap('hot');
clim([min(data,[],'all'),max(data,[],'all')]);
cb.Label.String = "Temperature (°C)";
set(gca,'ZDir','reverse');
grid on;
view(35,45);
% Slider positions

sliderWidth = 0.15; sliderHeight = 0.03; padding = 0.04;
% X-slice slider label
uicontrol('Style','text', 'String','X Slice', ...
    'Units','normalized', 'Position',[padding, padding+sliderHeight, sliderWidth, sliderHeight], ...
    'HorizontalAlignment','left');
% X-slice slider
uicontrol('Style','slider','Min',1,'Max',Nx,'Value',sx,'SliderStep',[1/(Nx-1), 10/(Nx-1)], ...
    'Units','normalized','Position',[padding padding sliderWidth sliderHeight], ...
    'Callback',@(src,evt) updateSlices(round(src.Value),1,hSliceX));

% Y-slice slider label
uicontrol('Style','text', 'String','Y Slice', ...
    'Units','normalized', 'Position',[padding, 2*padding+2*sliderHeight, sliderWidth, sliderHeight], ...
    'HorizontalAlignment','left');
% Y-slice slider
uicontrol('Style','slider','Min',1,'Max',Ny,'Value',sy,'SliderStep',[1/(Ny-1), 10/(Ny-1)], ...
    'Units','normalized','Position',[padding 2*padding+sliderHeight sliderWidth sliderHeight], ...
    'Callback',@(src,evt) updateSlices(round(src.Value),2,hSliceY));

% Z-slice slider label
uicontrol('Style','text', 'String','Z Slice', ...
    'Units','normalized', 'Position',[padding, 3*padding+3*sliderHeight, sliderWidth, sliderHeight], ...
    'HorizontalAlignment','left');
% Z-slice slider
uicontrol('Style','slider','Min',1,'Max',Nz,'Value',sz, 'SliderStep',[1/(Nz-1), 10/(Nz-1)],...
    'Units','normalized','Position',[padding 3*padding+2*sliderHeight sliderWidth sliderHeight], ...
    'Callback',@(src,evt) updateSlices(round(src.Value),3,hSliceZ));

% Nested function to update slices
    function updateSlices(ii,dim,slice)
        if dim == 1
            set(slice, 'XData', squeeze(X(ii,:,:)), 'YData', squeeze(Y(ii,:,:)), 'ZData', squeeze(Z(ii,:,:)), ...
                'CData', squeeze(data(ii,:,:)));
        elseif dim == 2
            set(slice, 'XData', squeeze(X(:,ii,:)), 'YData', squeeze(Y(:,ii,:)), ...
                'ZData', squeeze(Z(:,ii,:)), 'CData', squeeze(data(:,ii,:)));
        elseif dim == 3
            set(slice, 'CData', data(:,:,ii), 'XData', X(:,:,ii), 'YData', Y(:,:,ii), 'ZData', Z(:,:,ii));
        end
        drawnow;
    end

end