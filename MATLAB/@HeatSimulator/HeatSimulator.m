classdef HeatSimulator < handle
    %HEATSIMULATOR Summary of this class goes here
    %   Detailed explanation goes here

    properties
        mesh Mesh
        thermalInfo ThermalModel
        dt (1,1) double {mustBeGreaterThan(dt,0)} = 0.05 % Time Step for integration
        time (:,1) double
        alpha (1,1) double {mustBeInRange(alpha,0,1)} = 0.5 % implicit vs explicit lever
        sensorLocations (:,3) double
        silentMode (1,1) logical = false
        useAllCPUs (1,1) logical = false
        useGPU (1,1) logical = false
        resetIntegration (1,1) logical = true;
        buildMatrices (1,1) logical = true
    end

    methods
        function obj = HeatSimulator()
            %HEATSIMULATOR Construct an instance of this class
            %   Detailed explanation goes here
        end

        function [T,sensors] = solve(obj, finalTime, dt, alpha)
            arguments
                obj (1,1) HeatSimulator
                finalTime (1,1) {mustBeGreaterThan(finalTime,0)}
                dt double {mustBeGreaterThan(dt,0)} = []
                alpha double {mustBeInRange(alpha,0,1)} = []
            end
            if isempty(alpha)
                alpha = obj.alpha;
            end
            if isempty(dt)
                dt = obj.dt;
            end
            obj.dt = dt;
            obj.alpha = alpha;
            if (finalTime < obj.dt)
                error('Final time must be greater than the time step.');
            end
            obj.time = 0:obj.dt:finalTime;

            meshInfo = obj.mesh.toStruct();
            thermalInfoStruct = obj.thermalInfo.toStruct();
            settings = struct('finalTime',finalTime,'dt', obj.dt, 'alpha', obj.alpha,...
                'silentMode', obj.silentMode, 'useAllCPUs', obj.useAllCPUs, 'useGPU', obj.useGPU,...
                'resetIntegration',obj.resetIntegration,'sensorLocations',obj.sensorLocations);
            if (obj.buildMatrices)
                [T,sensors] = MEX_Heat_Simulation(thermalInfoStruct,settings,meshInfo);
            else
                [T,sensors] = MEX_Heat_Simulation(thermalInfoStruct,settings);
            end
            obj.thermalInfo.temperature = T;
        end

        function ax = plotSensorTemps(obj,sensorTemps)
            figure(1);
            clf;
            hold on;
            for ss = 1:size(obj.sensorLocations,1)
                plot(obj.time,sensorTemps(:,ss),'LineWidth',2,'DisplayName',...
                    sprintf("(%g,%g,%g)",obj.sensorLocations(ss,:)));
            end
            hold off
            grid on;
            xlabel("Time (s)");
            ylabel("Temperature (deg C)");
            title("Sensor temperature over time");
            legend()
            ax = gca;
        end

        function createVolumetricFigure(obj,figureNum)
            arguments
                obj
                figureNum = 2;
            end

            nodes = obj.mesh.nodes';
            elements = obj.mesh.elements;
            temp = obj.thermalInfo.temperature;

            % Initial ranges
            xrange = [min(nodes(:,1)) max(nodes(:,1))];
            yrange = [min(nodes(:,2)) max(nodes(:,2))];
            zrange = [min(nodes(:,3)) max(nodes(:,3))];

            % --- Create figure ---
            fig = figure(figureNum);
            clf(fig);
            fig.Name = 'Volumetric Temperature';

            % --- Create axes (right 3/4) ---
            ax = axes(fig,'Position',[0.3 0.1 0.65 0.85]);

            % --- Initial plot ---
            allFaces = Mesh.identifyAllFaces(elements);
            hPatch = plotVolumetricChunk(obj, allFaces, ax, xrange, yrange, zrange);

            xlabel(ax,'X Axis (cm)'); ylabel(ax,'Y Axis (cm)'); zlabel(ax,'Z Axis (cm)');
            grid(ax,'on'); colormap(ax,'hot'); colorbar(ax); view(ax,35,45); axis(ax,'equal');
            set(ax,'ZDir','reverse');

            % --- Left panel for sliders ---
            panel = uipanel(fig,'Title','Slice Controls','Position',[0 0 0.25 1]);  % left 25%

            sliderHeight = 0.07;   % fraction of panel height per slider
            spacing = 0.02;
            yPos = 0.9;  % start near top

            % Create sliders for X, Y, Z
            data = struct('obj', obj, 'ax', ax, 'faces', allFaces, ...
                'temp', temp, 'hPatch', hPatch, 'xrange', xrange, ...
                'yrange', yrange, 'zrange', zrange);

            fields = {'X','Y','Z'};
            for i = 1:3
                % Min slider
                uicontrol('Parent',panel,'Style','text','Units','normalized', ...
                    'Position',[0.05 yPos 0.4 0.04],'String',[fields{i} ' min']);
                sMin = uicontrol('Parent',panel,'Style','slider','Units','normalized', ...
                    'Position',[0.5 yPos 0.45 0.04], ...
                    'Min',min(nodes(:,i)),'Max',max(nodes(:,i)), ...
                    'Value',data.([lower(fields{i}) 'range'])(1), ...
                    'Callback', @(src,evt) HeatSimulator.sliderCallback(src,evt,[lower(fields{i}) 'min']));
                yPos = yPos - sliderHeight;

                % Max slider
                uicontrol('Parent',panel,'Style','text','Units','normalized', ...
                    'Position',[0.05 yPos 0.4 0.04],'String',[fields{i} ' max']);
                sMax = uicontrol('Parent',panel,'Style','slider','Units','normalized', ...
                    'Position',[0.5 yPos 0.45 0.04], ...
                    'Min',min(nodes(:,i)),'Max',max(nodes(:,i)), ...
                    'Value',data.([lower(fields{i}) 'range'])(2), ...
                    'Callback', @(src,evt) HeatSimulator.sliderCallback(src,evt,[lower(fields{i}) 'max']));
                yPos = yPos - (sliderHeight + spacing);

                % Save handles
                data.([fields{i} 'min']) = sMin;
                data.([fields{i} 'max']) = sMax;
            end

            guidata(fig,data);
        end
        function hPatch = plotVolumetricChunk(obj,elementFaces,ax,xrange,yrange,zrange)
            %PLOTVOLUME Plots volumetric temperature data
            %TODO: This function needs to be updated to plot arbitrary meshes.
            arguments (Input)
                obj
                elementFaces
                ax = gca
                xrange double = []
                yrange double = []
                zrange double = []
            end

            elements = obj.mesh.elements;
            nodes = obj.mesh.nodes;
            data = obj.thermalInfo.temperature;
            
            % ----- GET ALL FACES FOR EVERY ELEMENT --- %
            nNe = size(elements,1);
            nElem = size(elements,2);
            nFn = size(elementFaces,2);
            nFaces = size(elementFaces,1);
            
            % Compute bounding boxes for each face
            xMaxElem = max(reshape(nodes(1,elementFaces), nFaces, nFn), [], 2);  % max x per element
            xMinElem = min(reshape(nodes(1,elementFaces), nFaces, nFn), [], 2);
            yMaxElem = max(reshape(nodes(2,elementFaces), nFaces, nFn), [], 2);
            yMinElem = min(reshape(nodes(2,elementFaces), nFaces, nFn), [], 2);
            zMaxElem = max(reshape(nodes(3,elementFaces), nFaces, nFn), [], 2);
            zMinElem = min(reshape(nodes(3,elementFaces), nFaces, nFn), [], 2);

            % Keep only elements that intersect the ranges
            keep = (xMinElem <= xrange(2)) & (xMaxElem >= xrange(1)) & ...
                (yMinElem <= yrange(2)) & (yMaxElem >= yrange(1)) & ...
                (zMinElem <= zrange(2)) & (zMaxElem >= zrange(1));

            elementFaces = elementFaces(keep,:);
            [~, uniqueIdx, ic] = unique(sort(elementFaces, 2), 'rows'); % indices of unique faces
            elementFaces = elementFaces(uniqueIdx, :);      % preserve original node order
            counts = accumarray(ic, 1); % number of occurances for each face
            elementFaces = elementFaces(counts==1,:); % get only faces on boundary of region

            axis(ax);
            if isempty(ax.Children)
                hPatch = patch('Faces', elementFaces, ...
                    'Vertices', nodes', ... % need to tranpose nodes
                    'FaceVertexCData', data, ...
                    'FaceColor', 'interp', ...       % smooth coloring
                    'EdgeColor', 'none');
            else
                % reuse existing patch
                hPatch = ax.Children(1);
                set(hPatch, 'Faces', elementFaces);
            end
            
            hold off
            cb = colorbar;
            clim([min(data,[],'all'),max(data,[],'all')]);
        end
    end

    methods (Static)
        function sliderCallback(src, ~, type)
            fig = ancestor(src,'figure');
            data = guidata(fig);

            % Update ranges based on which slider was moved
            switch type
                case 'xmin', data.xrange(1) = src.Value;
                case 'xmax', data.xrange(2) = src.Value;
                case 'ymin', data.yrange(1) = src.Value;
                case 'ymax', data.yrange(2) = src.Value;
                case 'zmin', data.zrange(1) = src.Value;
                case 'zmax', data.zrange(2) = src.Value;
            end

            % Delete old patch and update
            % delete(data.hPatch); % patch is now updated directly in
            % function with set(hPatch, 'Faces', newFaces);
            data.hPatch = plotVolumetricChunk(data.obj, data.faces, ...
                data.ax, data.xrange, data.yrange, data.zrange);

            guidata(fig,data);
        end
    end
end