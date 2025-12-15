classdef Laser
    %LASER Summary of this class goes here
    %   Detailed explanation goes here

    properties
        waist (1,1) double {mustBeGreaterThan(waist,0)} = 1;
        wavelength (1,1) double {mustBeGreaterThan(wavelength,0)} = 10.6e-4;
        MUA (1,1) double {mustBeGreaterThanOrEqual(MUA,0)} = 0;
        MUS (1,1) double {mustBeGreaterThanOrEqual(MUS,0)} = 0;
        power (1,1) double {mustBeGreaterThanOrEqual(power,0)} = 1;
        focalPose (1,1) struct = struct('x',0,'y',0,'z',0,'theta',0,'phi',0,'psi',0)
        fluenceRate (:,1) double 
    end

    methods
        function obj = Laser(waist, wavelength, MUA)
            arguments
                waist (1,1) double = 1;
                wavelength (1,1) double = 10.6e-4;
                MUA (1,1) double = 1;
            end
            obj.waist = waist;
            obj.MUA = MUA;
            obj.wavelength = wavelength;
        end

        function obj = calculateIrradiance(obj,mesh)
            %METHOD1 Calculate the Irradiance of a laser incident on a
            %surface
            %   Assumes that the z location of the node in the mesh
            %   indicates the attenuation (absorption distance).
            %   Essentially, if the z = 0 location is not a flat surface,
            %   there will be errors
            
            arguments
                obj
                mesh (1,1) Mesh
            end
            % positions where we need to calculate fluence rate
            x = mesh.nodes(1,:);
            y = mesh.nodes(2,:);
            z = mesh.nodes(3,:);

            theta = obj.focalPose.theta; % Y rotation
            phi = obj.focalPose.phi; % X Rotation
            psi = obj.focalPose.psi; % Z Rotation
            RotMat = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)]*...  % Y Rotation
                [1 0 0; 0 cos(phi) -sin(phi); 0 sin(phi) cos(phi)]*...       % X Rotation
                [cos(psi) -sin(psi) 0; sin(psi) cos(psi) 0; 0 0 1];            % Z Rotation
            % Transformation is the laser frame with respect to tissue
            % frame 
            fx = obj.focalPose.x;
            fy = obj.focalPose.y;
            fz = obj.focalPose.z;
            Tmat = [RotMat [fx;fy;fz]; 0 0 0 1];
            % these are the locations on/in the tissue where we want the
            % irradiance, but in the laser's reference frame (optical axis reference
            % frame)
            p = Tmat\[x;y;z;ones(size(x))];
            % now calculate where the tissue surface is for absorption penetration
            % This assumes that the laser is emitted above the tissue surface
            % regardless of where the focal point is. 
            surface = Tmat\[x;y;zeros(size(z));ones(size(x))];
            
            % 
            % % Reshape back to matrices to make the output cleaner
            % Xt = reshape(p(1,:),size(X));
            % Yt = reshape(p(2,:),size(Y));
            % Zt = reshape(p(3,:),size(Z));
            % surfT = reshape(surface(3,:),size(Z));
            
            % calculate the beam waist using our coordinates wrt to the optical axis
            w = obj.waist * sqrt(1 + (p(3,:) .* obj.wavelength ./ (pi*obj.waist^2)).^2);
            % Calculate peak power density
            I0 = 2*obj.power ./ (w.^2 .* pi);
            obj.fluenceRate = I0.*exp(-2 .* (p(1,:).^2 + p(2,:).^2) ./ w.^2 - obj.MUA.*(p(3,:)-surface(3,:)));
        end
    end
end