classdef Mesh
    %MESH object to store nodes, elements, and boundary faces for use in
    %the FEM Heat Simulator
    %   Properties
    %       nodes - (3xn) vector with the x,y,z position of every node in
    %               the mesh
    %       elements - (Ne x e) vector with the nodes in each element
    %                   Ne is the number of nodes per element
    %                   e is the number of elements in the mesh
    %       boundaryFaces - (f x 1) struct with information on each
    %                       boundary face. Each face contains the following
    %                       fields:
    %                       elemID - element face belongs to
    %                       nodes - node indicies belonging to face
    %                       type - 0: flux, 1: convection, 2: heat sink
    %                       localFaceID - face number in element
    %
    %   The mesh object uses 1 indexing and the MEX file does the necessary
    %   conversion to 0 based indexing. For example, if we have a single 
    %   element mesh with eight nodes, the element will contain nodes
    %   [1,8]. Once passed into the MEX file, the c++ code will have an
    %   element with nodes [0,7]. This applies to node indicies, element
    %   indicies, localFaceIDs, etc. 
    %

    properties
        nodes (3,:) double
        elements (:,:) double
        boundaryFaces (:,1) struct
        useXYZ (1,1) logical = false;
        xpos (1,:) double
        ypos (1,:) double
        zpos (1,:) double
    end

    methods
        function obj = Mesh(varargin)
            if (nargin >= 1) && (nargin < 3)
                % assume we received a geometry object or femodel object
                % and potentially a list of boundaryTypes for each face in
                % the geometry
                if isa(varargin{1},'fegeometry')
                    geom = varargin{1};
                    femesh = geom.Mesh;
                elseif isa(varargin{1},'femodel')
                    geom = varargin{1}.Geometry;
                    femesh = geom.Mesh;
                end
                obj.nodes = femesh.Nodes;
                obj.elements = femesh.Elements;
                obj.boundaryFaces = Mesh.makeBoundaryFaces(femesh);

                % if two arguments were passed in then we also have boundary type
                if (nargin == 2) 
                    boundaryType = varargin{2};
                    if (length(boundaryType) < geom.NumFaces)
                        warning("Input 2 (boundaryType) only has %d entries but " + ...
                            "Geometry has %d faces. Padding with zeros",length(boundaryType),geom.NumFaces);
                        boundaryType = [boundaryType; zeros(geom.numFaces - length(boundaryType),1)];
                    elseif (length(boundaryType) > geom.NumFaces)
                        error("Input 2 (boundaryType) has %d entries but " + ...
                            "Geometry only has %d Faces.",length(boundaryType),geom.NumFaces);
                    end
                    obj.boundaryFaces = Mesh.classifyBoundaryFaces(obj.boundaryFaces,...
                        geom,boundaryType);
                end
            elseif (nargin == 3)
                obj.nodes = varargin{1};
                obj.elements = varargin{2};
                obj.boundaryFaces = varargin{3};
            elseif (nargin == 4)
                obj.xpos = varargin{1};
                obj.ypos = varargin{2};
                obj.zpos = varargin{3};
                boundaryConditions = varargin{4};
                obj = obj.buildCubeMesh(obj.xpos,obj.ypos,obj.zpos,boundaryConditions);
                obj.useXYZ = false;
            end
        end

        function S = toStruct(obj)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            S = struct();
            S.nodes = obj.nodes;
            S.elements = obj.elements;
            S.boundaryFaces = obj.boundaryFaces;
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

            % --- 1) Bottom face (z = 0), local face 1 ---
            [i,j] = ndgrid(1:Nx-1, 1:Ny-1);
            k = ones(size(i));
            obj.boundaryFaces = Mesh.makeSquareFaces(i,j,k, 1, bc(1), [1 4 3 2], Ny, Nx);
            % obj.boundaryFaces = [obj.boundaryFaces;BF];

            % --- 2) Top face (z = max), local face 2 ---
            [i,j] = ndgrid(1:Nx-1, 1:Ny-1);
            k = (Nz-1)*ones(size(i));
            BF = Mesh.makeSquareFaces(i,j,k, 2, bc(2), [5 6 7 8], Nx, Ny);
            obj.boundaryFaces = [obj.boundaryFaces;BF];


            % --- 3) Back face (y = min), local face 3 ---
            [i,k] = ndgrid(1:Nx-1, 1:Nz-1);
            j = ones(size(i));
            BF = Mesh.makeSquareFaces(i,j,k, 3, bc(3), [1 2 6 5], Nx, Nz);
            obj.boundaryFaces = [obj.boundaryFaces;BF];

            % --- 4) Front face (y = max), local face 4 ---
            [i,k] = ndgrid(1:Nx-1, 1:Nz-1);
            j = (Ny-1)*ones(size(i));
            BF = Mesh.makeSquareFaces(i,j,k, 4, bc(4), [4 8 7 3], Nz, Nx);
            obj.boundaryFaces = [obj.boundaryFaces;BF];

            % --- 5) Left face (x = min), local face 5 ---
            [j,k] = ndgrid(1:Ny-1, 1:Nz-1);
            i = ones(size(j));
            BF = Mesh.makeSquareFaces(i,j,k, 5, bc(5), [1 5 8 4], Nz, Nx);
            obj.boundaryFaces = [obj.boundaryFaces;BF];

            % --- 6) Right face (x = max), local face 6 ---
            [j,k] = ndgrid(1:Ny-1, 1:Nz-1);
            i = (Nx-1)*ones(size(j));
            BF = Mesh.makeSquareFaces(i,j,k, 6, bc(6), [2 3 7 6], Nz, Nx);
            obj.boundaryFaces = [obj.boundaryFaces;BF];

            fprintf(" Done.");
            toc

        end
    end
    methods(Static)
        function BF = makeSquareFaces(i, j, k, localFaceID, bcType, nodePattern, Nx, Ny)
            arguments
                i (:,:) double
                j (:,:) double
                k (:,:) double
                localFaceID (1,1) double
                bcType (1,1) double
                nodePattern (1,4) double
                Nx (1,1) double
                Ny (1,1) double
            end
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

        function boundaryFaces = makeBoundaryFaces(mesh)
            arguments
                mesh (1,1) pde.FEMesh
            end
            % makeBoundaryFaces  Construct boundary face data for a quadratic tet mesh.
            %
            %   boundaryFaces = makeBoundaryFaces(mesh)
            %
            %   mesh.Elements must be 10xN (quadratic tetrahedral elements)
            %   mesh.Nodes    must be 3xM
            %
            %   Output fields:
            %       elementID   - tetra index in mesh.Elements
            %       localFaceID - which of the 4 local faces (1..4)
            %       nodeIDs     - 6 node IDs defining that quadratic face
            %       type        - default = 0 (user can assign later)

            elements = mesh.Elements;     % 10 x Ne
            Ne = size(elements,2);

            % --- Quadratic tet face-node patterns ---
            faceNodePattern = {
                [1, 2, 3, 5, 9, 8]+1;   % face 1: Opposite node 1
                [0, 3, 2, 7, 9, 6]+1;   % face 2: Opposite node 2
                [0, 1, 3, 4, 8, 7]+1;   % face 3: Opposite node 3
                [0, 2, 1, 6, 5, 4]+1    % face 4: Opposite node 4
                };

            % Preallocate max possible faces (4 per element)
            allFaces(Ne*4).nodes = [];
            idx = 1;

            % --- Enumerate faces for all elements ---
            for eid = 1:Ne
                elemNodes = elements(:,eid);
                for lf = 1:4
                    fn = elemNodes(faceNodePattern{lf});
                    allFaces(idx).nodes       = sort(fn(:));  % row, sorted for matching
                    allFaces(idx).elemID   = eid;
                    allFaces(idx).localFaceID = lf;
                    idx = idx + 1;
                end
            end

            % Trim unused prealloc (in case)
            allFaces = allFaces(1:(idx-1));

            % --- Create hash keys for identifying repeated faces ---
            keys = cellfun(@(x) sprintf('%d_', x), {allFaces.nodes}, 'UniformOutput', false);

            % Unique face sets
            [uniqueKeys, ~, ic] = unique(keys);
            counts = accumarray(ic, 1); % count the number of times each unique key appears

            % Allocate boundaryFaces (worst case = all faces)
            boundaryFaces = struct('elemID',{},'localFaceID',{},'nodes',{},'type',{});
            b = 1;

            % --- Extract faces that occur exactly once (boundary) ---
            for k = 1:numel(uniqueKeys)
                if counts(k) == 1
                    idxFace = ic == k;
                    f = allFaces(idxFace); % Using logical indexing

                    boundaryFaces(b).elemID      = f.elemID;
                    boundaryFaces(b).localFaceID = f.localFaceID;
                    boundaryFaces(b).nodes       = f.nodes;
                    boundaryFaces(b).type        = 0;   % default user boundary type
                    b = b + 1;
                end
            end
        end

        function boundaryFaces = classifyBoundaryFaces(boundaryFaces, geometry, geomFaceType)
            % classifyBoundaryFaces
            %
            %   boundaryFaces = classifyBoundaryFaces(mesh, geometry, geomFaceType)
            %
            %   INPUTS:
            %       boundaryFaces : boundary faces taken from mesh object
            %                       (10-node quadratic tetrahedra)
            %       geometry      : PDE Geometry object (with NumFaces)
            %       geomFaceType  : vector of size geometry.NumFaces, where
            %                       geomFaceType(f) = type assigned to geometry face f
            %
            %   OUTPUT: -- updated type for each boundary face
            %       boundaryFaces(i).elementID
            %       boundaryFaces(i).localFaceID
            %       boundaryFaces(i).nodeIDs     (6 node IDs)
            %       boundaryFaces(i).type        (assigned BC type)
            %       boundaryFaces(i).geomFaceID  (which geometry face it lies on)
            %
            %   ----------------------------------------------------------------------
            %   This function performs three tasks:
            %       2) Map each boundary face → PDE geometry face
            %       3) Assign user-defined boundary type
            %   ----------------------------------------------------------------------
            arguments
                boundaryFaces (:,1) struct
                geometry (1,1) fegeometry
                geomFaceType (:,1)
            end
            %% ----------------------------------------------------
            % 1. Build geometry-face → node mapping (fast lookup)
            % ----------------------------------------------------
            geomFaceNodes = cell(geometry.NumFaces,1);
            for gf = 1:geometry.NumFaces
                geomFaceNodes{gf} = boundaryNode(mesh, 'region', gf);
            end
            %% ----------------------------------------------------------
            % 2. Assign geometry-face ID + boundary type to each face
            % ----------------------------------------------------------
            for k = 1:numel(boundaryFaces)
                % Use vertex nodes (first 3 nodes) for classification
                vtx = boundaryFaces(k).nodeIDs(1:3);

                found = false;
                for gf = 1:geometry.NumFaces
                    if all(ismember(vtx, geomFaceNodes{gf}))
                        boundaryFaces(k).geomFaceID = gf;
                        boundaryFaces(k).type = geomFaceType(gf);
                        found = true;
                        break;
                    end
                end

                if ~found
                    error('Could not classify boundary face k=%d into any geometry face.', k);
                end
            end

        end

    end
end