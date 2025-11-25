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

    end
end