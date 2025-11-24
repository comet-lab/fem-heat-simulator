classdef Mesh
    %MESH Summary of this class goes here
    %   Detailed explanation goes here

    properties
        xpos (1,:) double
        ypos (1,:) double
        zpos (1,:) double
        boundaryConditions (6,1)
        nodes (3,:) double
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
                obj = obj.buildCubeMesh(obj.xpos,obj.ypos,obj.zpos,obj.boundaryConditions);
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

        function obj = buildCubeMesh(obj, xPos, yPos, zPos, bc)

            Nx = numel(xPos);
            Ny = numel(yPos);
            Nz = numel(zPos);
            %% --- Build nodes ---
            fprintf("Building Nodes...")
            tic
            [Y, X, Z] = meshgrid(yPos, xPos, zPos);
            obj.nodes = [X(:)'; Y(:)'; Z(:)'];
            fprintf("... Done building nodes");
            toc

            fprintf("Building Elements...");
            tic

            % Element grid indices (lower-left-front corner of each voxel)
            [i, j, k] = ndgrid(1:Nx-1, 1:Ny-1, 1:Nz-1);

            i = i(:);
            j = j(:);
            k = k(:);

            % Compute base nodeID for each element
            base = (i-1) + (j-1)*Nx + (k-1)*Nx*Ny;   % corresponds to nodeID(i,j,k)-1

            % Offsets for the 8 nodes of each hexahedron (zero-based)
            offsets = [ ...
                0, ...
                1, ...
                1 + Nx, ...
                Nx, ...
                Nx*Ny, ...
                Nx*Ny + 1, ...
                Nx*Ny + 1 + Nx, ...
                Nx*Ny + Nx ...
                ];

            % Build all elements at once (8 × numElems)
            obj.elements = base.' + offsets.' + 1;

            fprintf("Done.");
            toc

            %% --- Build boundary faces ---
            obj.boundaryFaces = [];

            % --- 1) Bottom face (z = 0), local face 0 ---
            [i,j] = ndgrid(1:Nx-1, 1:Ny-1);
            k = ones(size(i));
            obj.boundaryFaces = Mesh.makeFaces(i,j,k, 0, bc(1), [1 4 3 2], Ny, Nx);
            % obj.boundaryFaces = [obj.boundaryFaces;BF];

            % --- 2) Top face (z = max), local face 1 ---
            [i,j] = ndgrid(1:Nx-1, 1:Ny-1);
            k = (Nz-1)*ones(size(i));
            BF = Mesh.makeFaces(i,j,k, 1, bc(2), [5 6 7 8], Nx, Ny);
            obj.boundaryFaces = [obj.boundaryFaces;BF];


            % --- 3) Back face (y = min), local face 2 ---
            [i,k] = ndgrid(1:Nx-1, 1:Nz-1);
            j = ones(size(i));
            BF = Mesh.makeFaces(i,j,k, 2, bc(3), [1 2 6 5], Nx, Nz);
            obj.boundaryFaces = [obj.boundaryFaces;BF];

            % --- 4) Front face (y = max), local face 3 ---
            [i,k] = ndgrid(1:Nx-1, 1:Nz-1);
            j = (Ny-1)*ones(size(i));
            BF = Mesh.makeFaces(i,j,k, 3, bc(4), [4 8 7 3], Nz, Nx);
            obj.boundaryFaces = [obj.boundaryFaces;BF];

            % --- 5) Left face (x = min), local face 4 ---
            [j,k] = ndgrid(1:Ny-1, 1:Nz-1);
            i = ones(size(j));
            BF = Mesh.makeFaces(i,j,k, 4, bc(5), [1 5 8 4], Nz, Nx);
            obj.boundaryFaces = [obj.boundaryFaces;BF];

            % --- 6) Right face (x = max), local face 5 ---
            [j,k] = ndgrid(1:Ny-1, 1:Nz-1);
            i = (Nx-1)*ones(size(j));
            BF = Mesh.makeFaces(i,j,k, 5, bc(6), [2 3 7 6], Nz, Nx);
            obj.boundaryFaces = [obj.boundaryFaces;BF];

            fprintf(" Done.");
            toc

        end
    end
    methods(Static)
        function BF = makeFaces(i, j, k, localFaceID, bcType, nodePattern, Nx, Ny)
            % Compute elem IDs
            elem = (i-1) + (j-1)*(Nx-1) + (k-1)*(Nx-1)*(Ny-1) + 1;

            % Compute 8 possible corner offsets
            NxNy = Nx*Ny;
            offset = [ ...
                0, ...               % (i,j,k)
                1, ...               % (i+1,j,k)
                Nx+1, ...            % (i+1,j+1,k)
                Nx, ...              % (i,j+1,k)
                NxNy, ...            % (i,j,k+1)
                NxNy+1, ...          % (i+1,j,k+1)
                NxNy+Nx+1, ...       % (i+1,j+1,k+1)
                NxNy+Nx              % (i,j+1,k+1)
                ];

            % Convert i,j,k into the base node index (zero-based)
            base = (i-1) + (j-1)*Nx + (k-1)*NxNy;

            % Build nodes for each face
            N = numel(elem);
            faceNodes = base(:).' + offset(nodePattern).' + 1;
            % Split into cell arrays
            elemC  = num2cell(elem(:));
            localC = num2cell(localFaceID * ones(N,1));
            typeC  = num2cell(bcType * ones(N,1));
            nodesC = squeeze(num2cell(faceNodes, 1)).';   % N×1 cells, each is 4×1

            % Build N×1 struct array
            BF = struct( ...
                'elemID',      elemC, ...
                'localFaceID', localC, ...
                'nodes',       nodesC, ...
                'type',        typeC ...
                );
        end
    end
end