classdef HeatSimulator < handle
    %HEATSIMULATOR This class controls the time stepping of the FEM
    %Simulator
    %   All information regarding running simulations should go through
    %   this class. For example, to set the fluence rate of the laser use
    %   
    %       simulator.thermalInfo.fluenceRate = ...
    %
    %   HeatSimulator is a handle class (not a value class). This means if
    %   you try to make a copy with
    %
    %       simulator2 = simulator1
    %       simulator2.thermalInfo.Temp = 0
    %
    %   both simulator objects will have the thermalInfo.temp = 0. Use the
    %   deepCopy() function to create a copy. Note that the Mesh object is
    %   also a handle object. Deep copy will not create a deep copy of the
    %   Mesh object and so even with
    %
    %       simulator2 = simulator1.deepCopy()
    %       simulator2.mesh.nodes = 0
    %
    %   simulator1 will have its mesh.nodes attribute altered.
    %
    %   Properties
    %       mesh Mesh - stores the mesh information
    %       thermalInfo ThermalModel - stores the thermalInformation of the
    %           mesh
    %       dt double - size of time step
    %       alpha double - implicit vs explicit integration lever
    %       time double - stores time points simulated with solve
    %       sensorLocations (:,3) double - stores the locations in the mesh
    %           that we want to get specific temperature information
    %       sensorTemps (:,:) double - sensor temperatures at each time
    %           point. This is populated after calling solve()
    %       silentMode logical - whether the MEX should print statements
    %       useAllCPUs logical - whether the simulator should use all the
    %           CPUs in the C++ code
    %       useGPU logical - whether the simulator should run on a GPU
    %       resetIntegration logical - setting to true will reset the time
    %           integration in the MEX file, essentially reseting the
    %           time derivative of our temperature
    %       buildMatrices logical - setting to true will have the MEX code
    %           build the global matrices from the mesh object. False will
    %           not pass a mesh struct into the MEX file.
    %
    %

    properties
        mesh Mesh
        thermalInfo ThermalModel
        laser Laser
        % Storage for time steping
        dt (1,1) double {mustBeGreaterThan(dt,0)} = 0.05 % Time Step for integration
        alpha (1,1) double {mustBeInRange(alpha,0,1)} = 0.5 % implicit vs explicit lever
        time (:,1) double = [0]
        % storage for sensor data
        sensorLocations (:,3) double
        sensorTemps (:,:) double
        % storage for miscellaneous settings
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

        function newObj = deepCopy(obj)
            newObj = HeatSimulator();
            newObj.mesh = obj.mesh;
            newObj.thermalInfo = obj.thermalInfo;
            newObj.laser = obj.laser;
            newObj.dt = obj.dt;
            newObj.alpha = obj.alpha;
            newObj.sensorTemps = obj.sensorTemps;
            newObj.time = obj.time;
            newObj.sensorLocations = obj.sensorLocations;
            newObj.silentMode = obj.silentMode;
            newObj.useAllCPUs = obj.useAllCPUs;
            newObj.useGPU = obj.useGPU;
            newObj.resetIntegration = obj.resetIntegration;
            newObj.buildMatrices = obj.buildMatrices;
        end

        function [T,sensorData,surfaceData] = solve(obj, timePoints, laserPose, laserPower)
            %SOLVE - solves the heat equation and returns temperature at
            %specific spatio-temporal coordiantes and the final temperature
            %at each node
            %   This function runs the heat equation for the duration
            %   specified in the timePoints input. It will return the
            %   temperature of the entire mesh after the specified duration
            %   as well as the temperature at each sensor location, at each
            %   time point specified in timePoints. The time step size of
            %   the simulator is dependent on dt. 
            %
            %   timePoints is expected to start at 0 and end at some
            %   nonzero time.
            %
            %   Additionally, this function will update the objects
            %   collective time and sensorTemps attributes if
            %   resetIntegration is false. This allows the user to call
            %   solve multiple times, and the object will track the
            %   temperature at the sensor locations across all solve calls.
            %
            %   [T,sensorData] = simulator.solve([0,1])
            %       this will return sensorData with 2 rows.
            arguments
                obj (1,1) HeatSimulator
                timePoints (:,1) {mustBeGreaterThanOrEqual(timePoints,0)}
                laserPose (:,6) = []
                laserPower (:,1) = []
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Conversion of objects to structs
            meshInfo = obj.mesh.toStruct();
            thermalInfoStruct = obj.thermalInfo.toStruct();
            settings = struct('time',timePoints,'dt', obj.dt, 'alpha', obj.alpha,...
                'silentMode', obj.silentMode, 'useAllCPUs', obj.useAllCPUs, 'useGPU', obj.useGPU,...
                'resetIntegration',obj.resetIntegration,'sensorLocations',obj.sensorLocations);

            if isempty(laserPose)
                laserStruct = struct('fluenceRate',obj.laser.fluenceRate);
            else
                laserStruct = struct('laserPose',laserPose,'laserPower',laserPower,...
                    'beamWaist',obj.laser.waist,'wavelength',obj.laser.wavelength);
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % running MEX file
            if (obj.buildMatrices)
                if nargout == 3
                    [T,sensorData,surfaceData] = MEX_Heat_Simulation(thermalInfoStruct,settings,laserStruct,meshInfo);
                else
                    [T,sensorData] = MEX_Heat_Simulation(thermalInfoStruct,settings,laserStruct,meshInfo);
                end
            else
                if nargout == 3
                    [T,sensorData,surfaceData] = MEX_Heat_Simulation(thermalInfoStruct,settings,laserStruct);
                else
                    [T,sensorData] = MEX_Heat_Simulation(thermalInfoStruct,settings,laserStruct);
                end
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Storing outputs
            obj.thermalInfo.temperature = T;

            if isscalar(timePoints)
                timePoints = [0;timePoints]; % just to make sure plotting is nice
            end
            if (obj.resetIntegration || obj.buildMatrices)
                obj.time = timePoints;
                obj.sensorTemps = sensorData;
            else
                obj.sensorTemps = [obj.sensorTemps(1:end-1,:); sensorData];
                obj.time = [obj.time(1:end-1); (timePoints+obj.time(end))];
            end

        end

        function createSensorTempsFigure(obj,figureNum)
            arguments
                obj
                figureNum = 1;
            end
            fig = figure(figureNum);
            clf(fig);
            fig.Name = 'SensorTemperatures';

            obj.plotSensorTemps();

            grid on;
            xlabel("Time (s)");
            ylabel("Temperature (deg C)");
            title("Sensor temperature over time");
            legend()
        end


        function ax = plotSensorTemps(obj,ax,plotOpts)
            arguments
                obj
                ax = gca;
                plotOpts.Marker = 'none'
                plotOpts.LineWidth = 2;
                plotOpts.LineStyle {mustBeMember(plotOpts.LineStyle,{'-','--',':','-.'})} = '-'
                plotOpts.ColorOrder (:,3) = colororder
            end
            % make sure we have enough colors and stack if we don't
            while size(obj.sensorLocations,1) > size(plotOpts.ColorOrder,1)
                plotOpts.ColorOrder = [plotOpts.ColorOrder;plotOpts.ColorOrder];
            end
            axis(ax);
            hold on;
            markerIndices = 1:ceil(length(obj.time)/20):length(obj.time);
            for ss = 1:size(obj.sensorLocations,1)
                plot(obj.time,obj.sensorTemps(:,ss),'LineWidth',plotOpts.LineWidth,...
                    'Marker',plotOpts.Marker,'LineStyle',plotOpts.LineStyle,...
                    'DisplayName',sprintf("Sensor@(%g,%g,%g) cm",obj.sensorLocations(ss,:)),...
                    'Color',plotOpts.ColorOrder(ss,:),'MarkerIndices',markerIndices);
            end
            hold off
            
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
            faceData = Mesh.identifyUniqueFaces(elements);
            hPatch = plotVolumetricChunk(obj, faceData, xrange, yrange, zrange, ax);

            xlabel(ax,'X Axis (cm)'); ylabel(ax,'Y Axis (cm)'); zlabel(ax,'Z Axis (cm)');
            grid(ax,'on'); colormap(ax,'hot'); colorbar(ax); view(ax,35,45); axis(ax,'equal');
            set(ax,'ZDir','reverse');

            % --- Left panel for sliders ---
            panel = uipanel(fig,'Title','Slice Controls','Position',[0 0 0.25 1]);  % left 25%

            sliderHeight = 0.07;   % fraction of panel height per slider
            spacing = 0.02;
            yPos = 0.9;  % start near top

            % Create sliders for X, Y, Z
            data = struct('obj', obj, 'ax', ax, 'faces', faceData, ...
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
        function hPatch = plotVolumetricChunk(obj,faceData,xrange,yrange,zrange,ax)
            %PLOTVOLUME Plots volumetric temperature data
            %TODO: This function needs to be updated to plot arbitrary meshes.
            arguments (Input)
                obj
                faceData
                xrange (1,2) double
                yrange (1,2) double
                zrange (1,2) double
                ax = gca
            end

            nodes = obj.mesh.nodes;
            data = obj.thermalInfo.temperature;
            facesToPlot = obj.sortBoundaryFaces(faceData,xrange,yrange,zrange);

            axis(ax);
            if isempty(ax.Children)
                hPatch = patch('Faces', facesToPlot, ...
                    'Vertices', nodes', ... % need to tranpose nodes
                    'FaceVertexCData', data, ...
                    'FaceColor', 'interp', ...       % smooth coloring
                    'EdgeColor', 'none');
            else
                % reuse existing patch
                hPatch = ax.Children(1);
                set(hPatch, 'Faces', facesToPlot);
            end

            hold off
            clim([min(data,[],'all'),max(data,[],'all')]);
        end


        function boundaryFaces = sortBoundaryFaces(obj,faceData,xrange,yrange,zrange)
            %SORTBOUNDARYFACES - takes a list of face data and a cuboid
            %region in x,y,z to produce faces that are on the boundary of
            %the region for plotting
            % ----- GET ALL FACES FOR EVERY ELEMENT --- %
            nodes = obj.mesh.nodes;
            elements = obj.mesh.elements;
            nElem = size(elements,2);
            nNe = size(elements,1);

            % Compute bounding boxes for each face
            xMaxElem = max(reshape(nodes(1,elements), nNe, nElem), [], 1);  % max x per element
            xMinElem = min(reshape(nodes(1,elements), nNe, nElem), [], 1);
            yMaxElem = max(reshape(nodes(2,elements), nNe, nElem), [], 1);
            yMinElem = min(reshape(nodes(2,elements), nNe, nElem), [], 1);
            zMaxElem = max(reshape(nodes(3,elements), nNe, nElem), [], 1);
            zMinElem = min(reshape(nodes(3,elements), nNe, nElem), [], 1);

            % Keep only elements that intersect the ranges
            inSlice = (xMinElem <= xrange(2)) & (xMaxElem >= xrange(1)) & ...
                (yMinElem <= yrange(2)) & (yMaxElem >= yrange(1)) & ...
                (zMinElem <= zrange(2)) & (zMaxElem >= zrange(1));

            fte = faceData.elemID;  % nFaces x 2 (0 if absent)
            elem1 = fte(:,1);
            elem2 = fte(:,2);
            in1 = false(size(elem1)); in2 = false(size(elem2));
            in1(elem1>0) = inSlice(elem1(elem1>0));
            in2(elem2>0) = inSlice(elem2(elem2>0));

            mask = (in1 & ~in2) | (~in1 & in2) | (elem2==0 & in1);

            boundaryFaces = faceData.faces(mask, :);
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
               data.xrange, data.yrange, data.zrange, data.ax);

            guidata(fig,data);
        end

        function compareSimSensors(figNum,simObjs)
            arguments
                figNum double = []
            end
            arguments (Repeating)
                simObjs HeatSimulator
            end

            if isempty(figNum)
                figNum = 1;
            end
            if ~isscalar(figNum)
                error("figNum should be a scalar")
            end
            figure(figNum);
            clf;
            tiledlayout;
            nexttile();
            nSims = numel(simObjs);     % number of repeated simObj inputs
            legendText = {};
            markers = ["+","o","*",".","x","s","d","^","v",">","<","p","h","none"];
            for i = 1:nSims
                sim = simObjs{i};       % each element is a HeatSimulator
                sim.plotSensorTemps('Marker',markers(i));
                for ss = 1:size(sim.sensorLocations,1)
                    legendText = [legendText sprintf("Sim %d Sensor %d",i,ss)];
                end
            end
            grid on;
            title("Comparison of Sensor Data From Different Simulators");
            xlabel("Time (s)")
            ylabel("Temperature (°C)")
            leg = legend(legendText,'NumColumns',nSims);
            leg.Layout.Tile = 'east';
            set(gca,'FontSize',15)
        end
    end
end