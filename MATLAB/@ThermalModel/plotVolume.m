function h = plotVolume(obj)
%PLOTVOLUME Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    obj
end

arguments (Output)
    h
end

numNodes = length(obj.mesh.nodes);
X = ones(numNodes,1);
Y = ones(numNodes,1);
Z = ones(numNodes,1);
end