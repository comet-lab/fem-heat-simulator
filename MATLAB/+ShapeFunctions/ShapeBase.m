% +ShapeFunctions/ShapeBase.m
classdef (Abstract) ShapeBase
    methods (Abstract)
        N(obj, xi)      % shape values (Nx1) for one xi (or NxM for multiple)
        dN(obj, xi)     % shape grads in reference coords (NxD or NxD x M)
        info(obj)       % metadata struct: numNodes, nodeCoords, ordering
        insideRef(obj,xi)
    end
    methods
        function s = toStruct(obj)
            s.info = obj.info();
            s.N = @(xi) obj.N(xi);
            s.dN = @(xi) obj.dN(xi);
        end
    end
end