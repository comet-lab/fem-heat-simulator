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
            elements = obj.mesh.elements';
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
            hPatch = plotVolumetricChunk(obj, ax, xrange, yrange, zrange);

            xlabel(ax,'X Axis (cm)'); ylabel(ax,'Y Axis (cm)'); zlabel(ax,'Z Axis (cm)');
            grid(ax,'on'); colormap(ax,'hot'); colorbar(ax); view(ax,35,45); axis(ax,'equal');
            set(ax,'ZDir','reverse');

            % --- Left panel for sliders ---
            panel = uipanel(fig,'Title','Slice Controls','Position',[0 0 0.25 1]);  % left 25%

            sliderHeight = 0.07;   % fraction of panel height per slider
            spacing = 0.02;
            yPos = 0.9;  % start near top

            % Create sliders for X, Y, Z
            data = struct('obj', obj, 'ax', ax, 'nodes', nodes, 'elements', elements, ...
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
        function hPatch = plotVolumetricChunk(obj,ax,xrange,yrange,zrange)
            %PLOTVOLUME Plots volumetric temperature data
            %TODO: This function needs to be updated to plot arbitrary meshes.
            arguments (Input)
                obj
                ax = gca
                xrange double = []
                yrange double = []
                zrange double = []
            end

            elements = obj.mesh.elements';     % nElem x nNe tranposed for patch
            nodes = obj.mesh.nodes'; % tranposed to be n x 3 for patch
            data = obj.thermalInfo.temperature;

            firstTimeFlag = false;
            if isempty(xrange) || isempty(yrange) || isempty(zrange)
                firstTimeFlag = true;
                xrange = [min(nodes(:,1)) max(nodes(:,1))];
                yrange = [min(nodes(:,2)) max(nodes(:,2))];
                zrange = [min(nodes(:,3)) max(nodes(:,3))];
            end

            % ----- GET ALL FACES FOR EVERY ELEMENT --- %
            nElem = size(elements,1);
            nNe = size(elements,2);

            % Compute bounding boxes for each element
            xMaxElem = max(reshape(nodes(elements(:),1), nElem, nNe), [], 2);  % max x per element
            xMinElem = min(reshape(nodes(elements(:),1), nElem, nNe), [], 2);
            yMaxElem = max(reshape(nodes(elements(:),2), nElem, nNe), [], 2);
            yMinElem = min(reshape(nodes(elements(:),2), nElem, nNe), [], 2);
            zMaxElem = max(reshape(nodes(elements(:),3), nElem, nNe), [], 2);
            zMinElem = min(reshape(nodes(elements(:),3), nElem, nNe), [], 2);

            % Keep only elements that intersect the ranges
            keep = (xMinElem <= xrange(2)) & (xMaxElem >= xrange(1)) & ...
                (yMinElem <= yrange(2)) & (yMaxElem >= yrange(1)) & ...
                (zMinElem <= zrange(2)) & (zMaxElem >= zrange(1));

            elementsSlice = elements(keep,:);
            nElem = size(elementsSlice,1);
            nNe = size(elementsSlice,2);

            if nNe == 10
                nNf = 6;
                % --- Quadratic tet face-node patterns ---
                faceNodePattern = {
                    [1, 2, 3, 5, 9, 8]+1;   % face 1: Opposite node 1
                    [0, 3, 2, 7, 9, 6]+1;   % face 2: Opposite node 2
                    [0, 1, 3, 4, 8, 7]+1;   % face 3: Opposite node 3
                    [0, 2, 1, 6, 5, 4]+1    % face 4: Opposite node 4
                    };
            elseif nNe == 8
                % --- Linear Hex face-node patterns ---
                % add one for 1 indexing
                nNf = 4;
                faceNodePattern = {
                    [0, 3, 2, 1]+1;  % z = -1
                    [4, 5, 6, 7]+1;  % z = 1
                    [0, 1, 5, 4]+1;  % y = -1
                    [3, 7, 6, 2]+1;  % y = 1
                    [0, 4, 7, 3]+1;  % x = -1
                    [1, 2, 6, 5]+1  % x = 1
                    };
            end

            % Preallocate max possible faces
            nFaces = size(faceNodePattern,1);
            facesAll = zeros(nElem * nFaces, nNf);
            % --- Enumerate faces for all elements ---
            row = 1;
            for k = 1:nElem
                for f = 1:nFaces
                    % extract appropriate nodes idxs from element
                    facesAll(row, :) = elementsSlice(k,faceNodePattern{f,:});
                    row = row + 1;
                end
            end
            facesKey = sort(facesAll, 2);             % only for comparison
            [~, uniqueIdx] = unique(facesKey, 'rows'); % indices of unique faces
            facesUnique = facesAll(uniqueIdx, :);      % preserve original node order

            axis(ax);
            hPatch = patch('Faces', facesUnique, ...
                'Vertices', nodes, ...
                'FaceVertexCData', data, ...
                'FaceColor', 'interp', ...       % smooth coloring
                'EdgeColor', 'none');
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
            delete(data.hPatch);
            data.hPatch = plotVolumetricChunk(data.obj, data.ax, ...
                data.xrange, data.yrange, data.zrange);

            guidata(fig,data);
        end
    end
end