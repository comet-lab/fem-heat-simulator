classdef Mesh
    %MESH Summary of this class goes here
    %   Detailed explanation goes here

    properties
        xpos (1,:) double
        ypos (1,:) double
        zpos (1,:) double 
        boundaryConditions (6,1)
        nodes (:,1) double
        elements (:,:) double
        boundaryFaces (:,1) struct
        useXYZ (1,1) logical = false;
    end

    methods
        function obj = Mesh(varargin)
            if (nargin == 3)
                obj.nodes = varargin(1);
                obj.elements = varargin(2);
                obj.boundaryFaces = varargin(3);
            elseif (nargin == 4)
                obj.xpos = varargin{1};
                obj.ypos = varargin{2};
                obj.zpos = varargin{3};
                obj.boundaryConditions = varargin{4};
                obj.useXYZ = true;
            end
        end

        function S = toStruct(obj)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            S = struct();
            if obj.useXYZ
                S.xpos = obj.xpos;
                S.ypos = obj.ypos;
                S.zpos = obj.zpos;
                S.boundaryConditions = obj.boundaryConditions;
            else
                S.nodes = obj.nodes;
                S.elements = obj.elements;
                S.boundaryFaces = obj.boundaryFaces;
            end
        end
    end
end