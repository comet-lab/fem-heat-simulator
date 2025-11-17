#pragma once
#include <array>
#include <vector>
#include <Eigen/Dense>
//#include "../Mesh.hpp"  // your mesh/nodes definition

namespace ShapeFunctions {

    struct HexLinear {
        static constexpr int nNodes = 8;
        static constexpr int nGP = 8;
        static constexpr int nFaceGP = 4;
        static constexpr int nFaceNodes = 4;

        /*
        * Origin Reference Frame and shape formation
        *           * ---- x     0 --- 1
        *         / |           /|    /|
        *        /  |          2 --- 3 |
        *       y   z          | 4 --| 5
        *                      |/    |/
        *                      6 --- 7
        */
        static constexpr std::array<std::array<int, 4>, 6> faceConnectivity = { {
            {0, 1, 3, 2}, // top face
            {4, 5, 7, 6},  // bottom face
            {0, 1, 5, 4}, // back face
            {2, 3, 7, 6}, // front face
            {0, 2, 6, 4}, // Left face
            {1, 3, 7, 5} // right face
        } };

        // Evaluate shape function
        static inline float N(const std::array<float, 3>& xi, int A)
        {
            const float x = xi[0], y = xi[1], z = xi[2];
            switch (A) {
            case 0: return 0.125f * (1 - x) * (1 - y) * (1 - z);
            case 1: return 0.125f * (1 + x) * (1 - y) * (1 - z);
            case 2: return 0.125f * (1 - x) * (1 + y) * (1 - z);
            case 3: return 0.125f * (1 + x) * (1 + y) * (1 - z);
            case 4: return 0.125f * (1 - x) * (1 - y) * (1 + z);
            case 5: return 0.125f * (1 + x) * (1 - y) * (1 + z);
            case 6: return 0.125f * (1 - x) * (1 + y) * (1 + z);
            case 7: return 0.125f * (1 + x) * (1 + y) * (1 + z);
            default: return 0.0f;
            }
        }

        static inline float N_face(const std::array<float, 2>& gp, int nodeIdx, int face)
        {
            // A will be a value between 0 and 3. Face will be a value between 0 and 5
            int A = faceConnectivity[face][nodeIdx];
            std::array<float, 3> xi = mapFaceGPtoXi(gp, face);
            return N(xi, A);
        }

        static inline Eigen::Vector3f dNdxi(const std::array<float, 3>& xi, int A)
        {
            const float x = xi[0], y = xi[1], z = xi[2];
            Eigen::Vector3f dN;
            switch (A) {
            case 0: dN << -0.125f * (1 - y) * (1 - z), -0.125f * (1 - x) * (1 - z), -0.125f * (1 - x) * (1 - y); break;
            case 1: dN << 0.125f * (1 - y) * (1 - z), -0.125f * (1 + x) * (1 - z), -0.125f * (1 + x) * (1 - y); break;
            case 2: dN << -0.125f * (1 + y) * (1 - z), 0.125f * (1 - x) * (1 - z), -0.125f * (1 - x) * (1 + y); break;
            case 3: dN << 0.125f * (1 + y) * (1 - z), 0.125f * (1 + x) * (1 - z), -0.125f * (1 + x) * (1 + y); break;
            case 4: dN << -0.125f * (1 - y) * (1 + z), -0.125f * (1 - x) * (1 + z), 0.125f * (1 - x) * (1 - y); break;
            case 5: dN << 0.125f * (1 - y) * (1 + z), -0.125f * (1 + x) * (1 + z), 0.125f * (1 + x) * (1 - y); break;
            case 6: dN << -0.125f * (1 + y) * (1 + z), 0.125f * (1 - x) * (1 + z), 0.125f * (1 - x) * (1 + y); break;
            case 7: dN << 0.125f * (1 + y) * (1 + z), 0.125f * (1 + x) * (1 + z), 0.125f * (1 + x) * (1 + y); break;
            default: dN.setZero(); break;
            }
            return dN;
        }

        static inline Eigen::Vector2f dNdxi_face(const std::array<float, 2>& gp, int nodeIdx, int face)
        {
            int A = faceConnectivity[face][nodeIdx];

            // Construct the full xi for the element, with the fixed coordinate for this face
            std::array<float, 3> xi = mapFaceGPtoXi(gp, face);

            // Compute full 3D derivative in reference coordinates
            Eigen::Vector3f dN = dNdxi(xi, A);

            // Select the two derivatives corresponding to the face parametric directions
            Eigen::Vector2f dN_face;
            switch (face)
            {
            case 0: case 1: dN_face << dN[0], dN[1]; break;
            case 2: case 3: dN_face << dN[0], dN[2]; break;
            case 4: case 5: dN_face << dN[1], dN[2]; break;
            }

            return dN_face;
        }

        static inline std::array<float, 3> mapFaceGPtoXi(const std::array<float, 2>& gp, int face)
        {
            std::array<float, 3> xi;
            switch (face)
            {
            case 0: xi[0] = gp[0]; xi[1] = gp[1]; xi[2] = -1.0f; break; // top
            case 1: xi[0] = gp[0]; xi[1] = gp[1]; xi[2] = 1.0f; break;  // bottom
            case 2: xi[0] = gp[0]; xi[1] = -1.0f; xi[2] = gp[1]; break; // back
            case 3: xi[0] = gp[0]; xi[1] = 1.0f; xi[2] = gp[1]; break;  // front
            case 4: xi[0] = -1.0f; xi[1] = gp[0]; xi[2] = gp[1]; break; // left
            case 5: xi[0] = 1.0f; xi[1] = gp[0]; xi[2] = gp[1]; break;  // right
            default: throw std::runtime_error("Invalid face index in mapFaceGPtoXi");
            }
            return xi;
        }

        // Gauss points and weights
        static inline std::vector<std::array<float, 3>> gaussPoints()
        {
            return {
                {-0.577350269f, -0.577350269f, -0.577350269f},
                { 0.577350269f, -0.577350269f, -0.577350269f},
                {-0.577350269f,  0.577350269f, -0.577350269f},
                { 0.577350269f,  0.577350269f, -0.577350269f},
                {-0.577350269f, -0.577350269f,  0.577350269f},
                { 0.577350269f, -0.577350269f,  0.577350269f},
                {-0.577350269f,  0.577350269f,  0.577350269f},
                { 0.577350269f,  0.577350269f,  0.577350269f}
            };
        }

        static inline std::vector<std::array<float, 2>> faceGaussPoints(int face)
        {
            // standard 2-point Gauss quadrature in 1D
            const float a = 1.0f / std::sqrt(3.0f);
            std::vector<std::array<float, 2>> gp(4);
            gp[0] = { -a, -a };
            gp[1] = {  a, -a };
            gp[2] = { -a,  a };
            gp[3] = {  a,  a };
            return gp;
        }

        static inline std::vector<std::array<float, 3>> weights()
        {
            return std::vector<std::array<float, 3>>(8, { 1.0,1.0,1.0 });
        }

        static inline std::vector<std::array<float, 2>> faceWeights(int face)
        {
            // 2D quad weights = product of 1D weights (both 1.0 for 2-point Gauss)
            std::vector<std::array<float, 2>> w(4);
            w[0] = { 1.0f, 1.0f };
            w[1] = { 1.0f, 1.0f };
            w[2] = { 1.0f, 1.0f };
            w[3] = { 1.0f, 1.0f };
            return w;
        }
    };
}