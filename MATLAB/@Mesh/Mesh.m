classdef Mesh < handle
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
                obj.nodes = femesh.Nodes; % convert m to cm
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
                xpos = varargin{1};
                ypos = varargin{2};
                zpos = varargin{3};
                boundaryConditions = varargin{4};
                obj = obj.buildCubeMesh(xpos,ypos,zpos,boundaryConditions);
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
            [Y, X, Z] = meshgrid(yPos, xPos, zPos);
            obj.nodes = [X(:)'; Y(:)'; Z(:)'];

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
            BF = Mesh.makeSquareFaces(i,j,k, 3, bc(3), [1 2 6 5], Nx, Ny);
            obj.boundaryFaces = [obj.boundaryFaces;BF];

            % --- 4) Front face (y = max), local face 4 ---
            [i,k] = ndgrid(1:Nx-1, 1:Nz-1);
            j = (Ny-1)*ones(size(i));
            BF = Mesh.makeSquareFaces(i,j,k, 4, bc(4), [4 8 7 3], Nx, Ny);
            obj.boundaryFaces = [obj.boundaryFaces;BF];

            % --- 5) Left face (x = min), local face 5 ---
            [j,k] = ndgrid(1:Ny-1, 1:Nz-1);
            i = ones(size(j));
            BF = Mesh.makeSquareFaces(i,j,k, 5, bc(5), [1 5 8 4], Nx, Ny);
            obj.boundaryFaces = [obj.boundaryFaces;BF];

            % --- 6) Right face (x = max), local face 6 ---
            [j,k] = ndgrid(1:Ny-1, 1:Nz-1);
            i = (Nx-1)*ones(size(j));
            BF = Mesh.makeSquareFaces(i,j,k, 6, bc(6), [2 3 7 6], Nx, Ny);
            obj.boundaryFaces = [obj.boundaryFaces;BF];

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
            %   mesh.Elements must be nNe x E
            %   mesh.Nodes    must be 3xN
            %
            %   Output fields:
            %       elementID   - tetra index in mesh.Elements
            %       localFaceID - which of the 4 local faces (1..4)
            %       nodeIDs     - 6 node IDs defining that quadratic face
            %       type        - default = 0 (user can assign later)

            elements = mesh.Elements;     % nNe x N

            faceData = Mesh.identifyUniqueFaces(elements);
            % faces on the boundary of the mesh only appear once
            uniqueIdx = faceData.counts == 1;
            % Extract boundary faces, elemID, and localFaceID
            faces = faceData.faces(uniqueIdx,:);
            elemIDs = faceData.elemID(uniqueIdx,1);
            localFaceIDs = faceData.localFaceID(uniqueIdx,1);


            % --- SET BOUNDARY FACES STRUCT ---
            boundaryFaces = struct('elemID',{},'localFaceID',{},'nodes',{},'type',{});
            % we have already isolated
            for k = 1:size(faces,1)
                boundaryFaces(k).elemID      = elemIDs(k);
                boundaryFaces(k).localFaceID = localFaceIDs(k);
                boundaryFaces(k).nodes       = faces(k,:);
                boundaryFaces(k).type        = 0;   % default user boundary type
            end
        end

        function faceData = identifyUniqueFaces(elements)
            %IDENTIFYALLFACES identifies all the faces in a Mesh
            %   Will not put duplicate faces. Any face shared by two
            %   elements will have that information stored in elemID and
            %   localFaceID
            %
            %   Inputs:
            %       elements: (nNe x nElems)
            %   Outputs:
            %       faceData: Struct with fields
            %           faces: (nE x nF) Contains a list of nodes corresponding
            %                  to each unique face
            %                  nE is the number of faces and
            %                  nF is the number of nodes on a face.
            %           elemID: (nE x 2) contains the element ID of each.
            %                   If a face belongs to more than one element,
            %                   the elemID will contain indicies for both
            %                   elements. Any value of 0 indiciates no
            %                   element
            %           localFaceID: (nE x 2) contains the local face numbering
            %                        for each face. Like elemID, if it
            %                        belongs to two elements it will
            %                        contain an second non-zero value
            %                        indiciating the local face id for the
            %                        second element.
            %           counts: (nE x 1) contains the number of times each face
            %                   appears in the list of elements
            %
            arguments
                elements (:,:)
            end

            nElem = size(elements,2);
            nNe = size(elements,1);

            if nNe == 10
                nNf = 6;
                % --- Quadratic tet face-node patterns ---
                faceNodePattern = {
                    % [1, 2, 3, 5, 9, 8]+1;   % face 1: Opposite node 1
                    % [0, 3, 2, 7, 9, 6]+1;   % face 2: Opposite node 2
                    % [0, 1, 3, 4, 8, 7]+1;   % face 3: Opposite node 3
                    % [0, 2, 1, 6, 5, 4]+1    % face 4: Opposite node 4
                    [1, 5, 2, 9, 3, 8]+1;   % face 1: Opposite node 1
                    [0, 7, 3, 9, 2, 6,]+1;   % face 2: Opposite node 2
                    [0, 4, 1, 8, 3, 7]+1;   % face 3: Opposite node 3
                    [0, 6, 2, 5, 1, 4]+1    % face 4: Opposite node 4
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
            allFaces = zeros(nElem * nFaces, nNf);
            elemID = zeros(nElem * nFaces,1);
            localFaceID = zeros(nElem * nFaces, 1);
            % --- Enumerate faces for all elements ---
            row = 1;
            for k = 1:nElem
                for f = 1:nFaces
                    % extract appropriate nodes idxs from element
                    allFaces(row, :) = elements(faceNodePattern{f,:},k);
                    elemID(row) = k;
                    localFaceID(row) = f;
                    row = row + 1;
                end
            end
            facesKey = sort(allFaces, 2);             % only for comparison
            % get unique faces
            [faceKey, ia, ic] = unique(facesKey, 'rows', 'stable');
            counts = accumarray(ic, 1); % number of occurances for each face
            nUnique = size(ia,1);
            faceToElem = zeros(nUnique,2); % faces can share two elements
            faceToFaceID = zeros(nUnique,2);

            % Some unique faces can be grabbed by a second index. The
            % variablce ic contains how to go from uniqueFace -> allFaces
            % so we will use that to find the second occurance of each face
            % in the unique face ID
            % We find which indicies from allFaces are not in ia which
            % means they are duplicates
            dupIdx = setdiff(1:size(facesKey,1), ia); % logical array to isolate faces with 2 elements
            % dupIdx contains the row in allFaces pertaining to a
            % duplicated face, we can now calculate the row in the
            % uniqueFaces by using ic
            dupRows = ic(dupIdx);   % rows in allFaces that belong to a repeated face

            % Now fill in the output arrays
            uniqueFaces = allFaces(ia,:);
            faceToElem(:,1)   = elemID(ia);
            faceToElem(dupRows,2)   = elemID(dupIdx);
            faceToFaceID(:,1) = localFaceID(ia);
            faceToFaceID(dupRows,2) = localFaceID(dupIdx);

            % package
            faceData.faces       = uniqueFaces;    % nFaces x nNodesPerFace
            faceData.elemID      = faceToElem;  % nFaces x 2
            faceData.faceKey     = faceKey;     % nFaces x nNodesPerFace (sorted)
            faceData.localFaceID = faceToFaceID;
            faceData.counts = counts;
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
                geomFaceNodes{gf} = findNodes(geometry.Mesh, 'region',"Face", gf);
            end
            %% ----------------------------------------------------------
            % 2. Assign geometry-face ID + boundary type to each face
            % ----------------------------------------------------------
            for k = 1:numel(boundaryFaces)
                % Use vertex nodes (first 3 nodes) for classification
                vtx = boundaryFaces(k).nodes;

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
        function x = geometricSpacing(a, b, n, sStart, flip)
            % Geometric spacing with fixed number of points
            % a, b      - start and end coordinates
            % n         - total number of points
            % sStart    - first spacing (largest)
            % flip      - optional, true to put largest spacing at b instead of a
            %
            % Output:
            % x         - vector of coordinates from a to b

            if nargin < 5
                flip = false;
            end

            if n < 2
                error('Number of points n must be at least 2');
            end

            L = b - a;        % total length
            m = n - 1;        % number of spacings

            % Handle r = 1 case (uniform spacing)
            if abs(sStart*m - L) < 1e-12
                r = 1;
            else
                % Solve geometric series equation for r
                fun = @(r) sStart*(1 - r^m)/(1 - r) - L;
                try
                    [r,~,exitflag,~] = fzero(fun, [0, 0.999]);  % r in (0,1)
                    if exitflag <= 0
                        error('GeometricSpacing:NoConverge','fzero failed: %s', output.message);
                    end
                catch ME
                    msg = sprintf("Most likely due to sStart*(n-1) < (b-a)");
                    error('fzero failed: %s\n%s', ME.message,msg);
                end
            end

            % Build spacings
            d = sStart * r.^(0:m-1);

            % Adjust slightly to hit exactly b
            scale = L / sum(d);
            d = d * scale;

            % Build coordinates
            x = a + [0, cumsum(d)];

            % Flip coordinates if needed (largest spacing at b)
            if flip
                x = b - (x - a);
                x = fliplr(x);  % optional to ensure ascending order
            end
        end

        function x = createGeometricSpacing(objLength,numNodes,hMax,hMin)
            % GETNODESPACING determines the geometric node spacing for a
            % specified length given the number of nodes and largest and
            % smallest spacing
            %
            % This function will keep appending chunks the size of hMin
            % until the spacing difference between each node are all >=
            % hMin and we have the correct number of nodes along the
            % length. This function will use geometricSpacing to determine
            % the intermediate spots
            %
            %
            arguments
                objLength (1,1) double {mustBeGreaterThan(objLength,0)}
                numNodes (1,1) double {mustBeGreaterThan(numNodes,1)}
                hMax (1,1) double {mustBeGreaterThan(hMax,0)}
                hMin (1,1) double {mustBeGreaterThanOrEqual(hMin,0)}
            end
            if hMax < hMin
                error("hMax must be greater than hMin")
            end
            valid = false;
            endNodes = objLength;
            tol = 1e-5;
            while ~valid
                remNodes = numNodes - length(endNodes) + 1;
                spacing = Mesh.geometricSpacing(0,endNodes(1),remNodes,hMax);
                if isscalar(endNodes)
                    x = spacing;
                else
                    x = [spacing endNodes(2:end)];
                end
                edgeLengths = diff(x);
                if any(abs(edgeLengths) < (hMin - tol))
                    newLeft = endNodes(1) - hMin;
                    endNodes = [newLeft endNodes];
                else
                    return
                end
            end
        end
    end
end