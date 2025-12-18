classdef Tet4 < ShapeFunctions.ShapeBase
    methods
        function obj = Tet4()
        end

        function Nvals = N(~, xi)
            % xi: 3xM with natural coords [xi; eta; zeta]
            % Reference tetra nodes: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
            xi1  = xi(1,:);   % 1xM
            eta  = xi(2,:);   % 1xM
            zeta = xi(3,:);   % 1xM

            Nvals = [ 1 - xi1 - eta - zeta;
                      xi1;
                      eta;
                      zeta ];
        end

        function dNvals = dN(~, xi)
            % dN for linear tetra (4-node)
            % xi: 3xM
            % dNvals: 4x3xM where columns correspond to d/dxi, d/deta, d/dzeta
            if isempty(xi), dNvals = zeros(4,3,0); return; end
            assert(size(xi,1)==3, 'xi must be 3xM');

            m = size(xi,2);

            % constant derivatives for linear tetra:
            % d/dxi:  [-1; 1; 0; 0]
            % d/deta: [-1; 0; 1; 0]
            % d/dzeta:[-1; 0; 0; 1]
            dxi   = [-1; 1; 0; 0];
            deta  = [-1; 0; 1; 0];
            dzeta = [-1; 0; 0; 1];

            dNvals = zeros(4,3,m);
            dNvals(:,1,:) = reshape(dxi,   [4,1,m]);
            dNvals(:,2,:) = reshape(deta,  [4,1,m]);
            dNvals(:,3,:) = reshape(dzeta, [4,1,m]);
        end

        function info = info(~)
            % Node coords in reference (xi,eta,zeta)
            % ordering matches N: 1:(0,0,0), 2:(1,0,0), 3:(0,1,0), 4:(0,0,1)
            info.numNodes  = 4;
            info.nodeCoords = [ 0 0 0;
                                1 0 0;
                                0 1 0;
                                0 0 1 ]; % 4x3
            info.name = 'linear tetrahedron (4-node)';
        end

        function isInside = insideRef(~,xi)
            arguments
                ~
                xi (3,:)
            end
            tol = 1e-6;
            isInside = all(xi >= (0-tol) & xi <= (1+tol), 1) & (sum(xi, 1) <= 1);
        end
    end
end
