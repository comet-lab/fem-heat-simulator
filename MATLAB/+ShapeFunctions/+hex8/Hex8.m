% +ShapeFunctions/+linhex/LinHex.m
classdef Hex8 < ShapeFunctions.ShapeBase
    methods
        function obj = Hex8()
        end

        function Nvals = N(~, xi)
            xi1 = xi(1,:); eta = xi(2,:); zeta = xi(3,:);
            Nvals = 1/8 * [ (1-xi1).*(1-eta).*(1-zeta);
                (1+xi1).*(1-eta).*(1-zeta);
                (1+xi1).*(1+eta).*(1-zeta);
                (1-xi1).*(1+eta).*(1-zeta);
                (1-xi1).*(1-eta).*(1+zeta);
                (1+xi1).*(1-eta).*(1+zeta);
                (1+xi1).*(1+eta).*(1+zeta);
                (1-xi1).*(1+eta).*(1+zeta) ];
        end
        
        function dNvals = dN(~, xi)
            % dN_linhex  Derivatives of 8-node hexahedral shape functions
            %   xi:  3xM array of natural coordinates [xi; eta; zeta]
            %   dN:  8x3xM array where dN(i,1,k)=dN_i/dxi, dN(i,2,k)=dN_i/deta,
            %        dN(i,3,k)=dN_i/dzeta for point k.
            %
            % Node ordering follows signs matrix below.

            % validate minimal shape (optional)
            if isempty(xi), dNvals = zeros(8,3,0); return; end
            assert(size(xi,1)==3, 'xi must be 3xM');

            % natural coordinates
            xi1  = xi(1,:);   % 1xM
            eta  = xi(2,:);   % 1xM
            zeta = xi(3,:);   % 1xM
            m = size(xi,2);

            % signs for nodes: [s_xi, s_eta, s_zeta]
            s = [-1 -1 -1;
                  1 -1 -1;
                  1  1 -1;
                 -1  1 -1;
                 -1 -1  1;
                  1 -1  1;
                  1  1  1;
                 -1  1  1];         % 8x3

            % factors: each is 8xM by implicit expansion (MATLAB R2016b+)
            fx = 1 + s(:,1) .* xi1;   % 8xM
            fy = 1 + s(:,2) .* eta;   % 8xM
            fz = 1 + s(:,3) .* zeta;  % 8xM

            % derivatives per analytic formula: dN_i/dxi = (s_xi/8) * (1+s_eta*eta)*(1+s_zeta*zeta)
            dN_dxi   = (s(:,1) / 8) .* (fy .* fz);   % 8xM
            dN_deta  = (s(:,2) / 8) .* (fx .* fz);   % 8xM
            dN_dzeta = (s(:,3) / 8) .* (fx .* fy);   % 8xM

            % assemble to 8x3xM
            dNvals = zeros(8,3,m);
            dNvals(:,1,:) = reshape(dN_dxi,   [8,1,m]);
            dNvals(:,2,:) = reshape(dN_deta,  [8,1,m]);
            dNvals(:,3,:) = reshape(dN_dzeta, [8,1,m]);
        end

        function info = info(~)
            s = [-1 -1 -1;
                  1 -1 -1;
                  1  1 -1;
                 -1  1 -1;
                 -1 -1  1;
                  1 -1  1;
                  1  1  1;
                 -1  1  1];         % 8x3
            info.numNodes = 8;
            info.nodeCoords = s; % reference corner coords
            info.name = 'linear hexahedron (8-node)';
        end

        function isInside = insideRef(~,xi)
            arguments
                ~
                xi (3,:)
            end
            tol = 1e-6;
            isInside = all(xi >= (-1-tol) & xi <= (1+tol), 1); % Check if all coordinates are within the reference element
        end
    end
end

