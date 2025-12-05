classdef ThermalModel
    %THERMALMODEL Currently just a container for variables needed in our
    %MEX function
    %   Detailed explanation goes here

    properties
        MUA (1,1) double {mustBeGreaterThan(MUA,0)} = 200 % absorption coefficient [cm^-1]
        TC (1,1) double {mustBeGreaterThan(TC,0)} = 0.006 % thermal conductivity [W/K cm]
        VHC (1,1) double {mustBeGreaterThan(VHC,0)} = 4.2 % volumetric heat capacity [J/K cm^3]
        HTC (1,1) double {mustBeGreaterThan(HTC,0)} = 0.01 % heat transfer coefficient [W/K cm^2]
        flux (1,1) double = 0 % flux value W/cm^2
        ambientTemp (1,1) double = 24 % ambient temp °C
        temperature (:,1) double
    end

    methods
        function obj = ThermalModel()
            %THERMALMODEL Construct an instance of this class
            %   Detailed explanation goes here
        end

        function S = toStruct(obj)
            p = properties(obj);
            S = struct();
            for k = 1:numel(p)
                S.(p{k}) = obj.(p{k});
            end
        end
    end
end