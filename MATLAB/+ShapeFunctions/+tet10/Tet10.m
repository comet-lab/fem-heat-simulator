classdef Tet10 < ShapeFunctions.ShapeBase
    methods
        function obj = Tet10()
        end

        function Nvals = N(~, xi)
            % N  Shape functions for quadratic tetrahedron (10-node)
            % xi: 3xM with natural coordinates [xi; eta; zeta]
            % Nvals: 10xM, ordering:
            % 1:(0,0,0)   corner L1
            % 2:(1,0,0)   corner L2
            % 3:(0,1,0)   corner L3
            % 4:(0,0,1)   corner L4
            % 5: edge 1-2 (mid)
            % 6: edge 2-3
            % 7: edge 3-1
            % 8: edge 1-4
            % 9: edge 2-4
            %10: edge 3-4

            assert(size(xi,1)==3, 'xi must be 3xM');
            xi1  = xi(1,:);   % L2
            eta  = xi(2,:);   % L3
            zeta = xi(3,:);   % L4

            L2 = xi1; L3 = eta; L4 = zeta;
            L1 = 1 - L2 - L3 - L4;   % barycentric L1

            % corner (quadratic) Ni = Li*(2*Li - 1)
            N1 = L1 .* (2*L1 - 1);
            N2 = L2 .* (2*L2 - 1);
            N3 = L3 .* (2*L3 - 1);
            N4 = L4 .* (2*L4 - 1);

            % edge mids Nij = 4 * Li * Lj
            N5  = 4 * L1 .* L2; % edge 1-2
            N6  = 4 * L2 .* L3; % edge 2-3
            N7  = 4 * L3 .* L1; % edge 3-1
            N8  = 4 * L1 .* L4; % edge 1-4
            N9  = 4 * L2 .* L4; % edge 2-4
            N10 = 4 * L3 .* L4; % edge 3-4

            Nvals = [N1;
                N2;
                N3;
                N4;
                N5;
                N6;
                N7;
                N8;
                N9;
                N10];
        end

        function dNvals = dN(~, xi)
            % dN  Vectorized derivatives of Tet10 shape functions
            % xi: 3xM
            % dNvals: 10x3xM (d/dxi, d/deta, d/dzeta)

            if isempty(xi), dNvals = zeros(10,3,0); return; end
            assert(size(xi,1)==3, 'xi must be 3xM');

            xi1  = xi(1,:);   % L2
            eta  = xi(2,:);   % L3
            zeta = xi(3,:);   % L4
            m = size(xi,2);

            L2 = xi1; L3 = eta; L4 = zeta;
            L1 = 1 - L2 - L3 - L4;
            L = [L1; L2; L3; L4];   % 4xM

            % constant derivatives of barycentrics: rows L1..L4, cols xi,eta,zeta
            dLdxi = [-1, -1, -1;
                1,  0,  0;
                0,  1,  0;
                0,  0,  1]; % 4x3

            % --- Corner nodes (i = 1..4) ---
            % Ni = Li*(2*Li - 1) => dNi/dx = (4*Li - 1) * dLi/dx
            % compute 4xM matrices for each component
            factor = 4 * L - 1;      % 4xM
            dN_c_x = factor .* dLdxi(:,1); % each row multiplied by scalar dLi/dxi -> 4xM
            dN_c_y = factor .* dLdxi(:,2); % 4xM
            dN_c_z = factor .* dLdxi(:,3); % 4xM

            % --- Edge nodes (pairs and formula) ---
            % edge Nij = 4 * Li * Lj
            % d(Nij)/dLi = 4*Lj, d(Nij)/dLj = 4*Li
            % d(Nij)/dx = 4 * (dLi * Lj + Li * dLj) for each component
            pairs = [1 2; 2 3; 3 1; 1 4; 2 4; 3 4];
            Li = L(pairs(:,1), :); % 6 x M
            Lj = L(pairs(:,2), :); % 6 x M
            dLi_x = dLdxi(pairs(:,1),1); dLj_x = dLdxi(pairs(:,2),1);
            dLi_y = dLdxi(pairs(:,1),2); dLj_y = dLdxi(pairs(:,2),2);
            dLi_z = dLdxi(pairs(:,1),3); dLj_z = dLdxi(pairs(:,2),3);

            dN_e_x = 4*(dLi_x .* Lj + dLj_x .* Li); % 6 x M
            dN_e_y = 4*(dLi_y .* Lj + dLj_y .* Li);
            dN_e_z = 4*(dLi_z .* Lj + dLj_z .* Li);

            % assemble into 10x3xM using reshape (no loops over M)
            dNvals = zeros(10,3,m);
            dNvals(1:4,1,:) = reshape(dN_c_x, [4,1,m]);
            dNvals(1:4,2,:) = reshape(dN_c_y, [4,1,m]);
            dNvals(1:4,3,:) = reshape(dN_c_z, [4,1,m]);

            dNvals(5:10,1,:) = reshape(dN_e_x, [6,1,m]);
            dNvals(5:10,2,:) = reshape(dN_e_y, [6,1,m]);
            dNvals(5:10,3,:) = reshape(dN_e_z, [6,1,m]);
        end

        function isInside = insideRef(~,xi)
            arguments
                ~
                xi (3,:)
            end
            tol = 1e-6;
            isInside = all(xi >= (0-tol) & xi <= (1+tol), 1) & (sum(xi, 1) <= 1);
        end

        function info = info(~)
            % Node coordinates in reference tetra (xi,eta,zeta)
            % ordering matches N:
            % 1:(0,0,0), 2:(1,0,0), 3:(0,1,0), 4:(0,0,1)
            % 5: mid(1-2), 6: mid(2-3), 7: mid(3-1), 8: mid(1-4), 9: mid(2-4), 10: mid(3-4)
            info.numNodes = 10;
            info.nodeCoords = [ 0   0   0;
                1   0   0;
                0   1   0;
                0   0   1;
                0.5 0   0;   % 1-2
                0.5 0.5 0;   % 2-3
                0   0.5 0;   % 3-1
                0   0   0.5; % 1-4
                0.5 0   0.5; % 2-4
                0   0.5 0.5];% 3-4
            info.name = 'quadratic tetrahedron (10-node)';
        end
    end
end
