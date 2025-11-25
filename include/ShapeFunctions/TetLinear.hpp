#pragma once
#include <array>
#include <vector>
#include <Eigen/Dense>
#include <stdexcept>

namespace ShapeFunctions {

    struct TetLinear {
        //EIGEN_MAKE_ALIGNED_OPERATOR_NEW


        /*
        *     zeta        3 
        *    /          / | \ 
        *   /          /  |  \
        *  *---> eta  0 --+-- 2
        *   \          \  |  /
        *    \          \ | /
        *     xi          1
        */

        /* Reference for shape functions and numerical integration Schemes: The finite element method: its basis and fundamentals (7th edition)
        * by O.C. Zienkiewicz, R.L. Taylor, J.Z. Zhu. Note that the weights in their tables are presumed to be multiplied by the volume of the 
        * element. In our case our element volume is 1/6 so we multiply the weights in the table by 1/6. The same is true for area where the area
        * of any face in the normalized coordinate frame is 1/2. 
        */
        static constexpr int nNodes = 4;
        static constexpr int nGP = 4; // the conductivity matrix only needs order 1, but M needs order 2
        static constexpr int nFaceGP = 3;   // 1-point quadrature per triangular face
        static constexpr int nFaceNodes = 3;
        static constexpr int nFaces = 4;

        // Face connectivity: 4 faces, 3 nodes each
        static constexpr std::array<std::array<int, 3>, 4> faceConnectivity = { {
            {1, 2, 3}, // opposite node 0 (xi + eta + zeta) = 1
            {0, 3, 2}, // opposite node 1 (xi = 0)
            {0, 1, 3}, // opposite node 2 (eta = 0)
            {0, 2, 1}  // opposite node 3 (zeta = 0)
        } };

        // Standard linear shape functions in reference (xi,eta,zeta)
        // Reference tetrahedron: 
        static inline float N(const std::array<float, 3>& xi, int A) {
            switch (A) {
            case 0: return 1.0f - xi[0] - xi[1] - xi[2]; // 1 if xi,eta,zeta = 0
            case 1: return xi[0]; // xi
            case 2: return xi[1]; // eta
            case 3: return xi[2]; // zeta
            default: return 0.0f;
            }
        }

        // Derivative of N wrt
        static inline Eigen::Vector3f dNdxi(const std::array<float, 3>&, int A) {
            // derivitives of shape functions are constant
            switch (A) {
            case 0: return Eigen::Vector3f(-1.0f, -1.0f, -1.0f);
            case 1: return Eigen::Vector3f(1.0f, 0.0f, 0.0f);
            case 2: return Eigen::Vector3f(0.0f, 1.0f, 0.0f);
            case 3: return Eigen::Vector3f(0.0f, 0.0f, 1.0f);
            default: return Eigen::Vector3f::Zero();
            }
        }

        // Face shape function (triangle in barycentric space)
        static inline float N_face(const std::array<float, 2>& xi, int A) {
            // Each triangular face uses local coords 
            switch (A) {
            case 0: return 1.0f - xi[0] - xi[1]; // 1 if xi,eta,zeta = 0
            case 1: return xi[0]; // xi
            case 2: return xi[1]; // eta
            default: return 0.0f;
            }
        }

        static inline Eigen::Vector2f dNdxi_face(const std::array<float, 2>& xi, int A) {
            switch (A) {
            case 0: return Eigen::Vector2f(-1.0f, -1.0f);
            case 1: return Eigen::Vector2f(1.0f, 0.0f);
            case 2: return Eigen::Vector2f(0.0f, 1.0f);
            default: return Eigen::Vector2f::Zero();
            }
        }

        // Map face Gauss point to element
        static inline std::array<float, 3> mapFaceGPtoXi(const std::array<float, 2>& gp, int face) {
            const float r = gp[0];
            const float s = gp[1];
            std::array<float, 3> xi{};
            switch (face) {
            case 0: xi = { 1.0f - r - s, r, s }; break; // {1, 2, 3} opposite node 0 (xi + eta + zeta) = 1
            case 1: xi = { 0, s, r }; break; //  {0, 3, 2} opposite node 1 (xi = 0)
            case 2: xi = { r, 0.0f, s }; break; //  {0, 1, 3} opposite node 2 (eta = 0)
            case 3: xi = { s, r, 0.0f }; break; // { 0, 2, 1} opposite node 3 (zeta = 0) 
            default: throw std::runtime_error("Invalid face index in TetLinear::mapFaceGPtoXi");
            }
            return xi;
        }

        // 1-point Gauss rule for tetrahedron
        static inline std::vector<std::array<float, 3>> gaussPoints() {
            //return { { {0.25f, 0.25f, 0.25f} } };  // centroid // single gauss point
            float alpha = 0.58541020f;
            float beta = 0.13819660f;
            return { { {alpha, beta, beta},
                {beta,alpha,beta},
                {beta,beta,alpha},
                {beta,beta,beta} } };
        }

        static inline std::vector<std::array<float, 3>> weights() {
            // volume: 1/6 for reference tetrahedral
            // return { { {1.0f, 1.0f, 1.0f} } };     // single point weight
            // 4-point weight
            // The weight isn't distributed to each (xi,eta,zeta) point. Its defined for the volume so I put
            // the volume scale factor as the first element and just set the others to 1. 
            return { { {1 / 24.0f, 1.0f, 1.0f}, 
                {1 / 24.0f, 1.0f, 1.0f},
                {1 / 24.0f, 1.0f, 1.0f},
                {1 / 24.0f, 1.0f, 1.0f},} };
        }

        // 1-point quadrature for triangular face (centroid)
        static inline std::vector<std::array<float, 2>> faceGaussPoints() {
            //return { { {1.0f / 3.0f, 1.0f / 3.0f} } }; // Single point quadrature
            return { { {1 / 2.0f, 1 / 2.0f},
                {1 / 2.0f, 0.0f},
                {0.0f, 1 / 2.0f} } };
        }

        static inline std::vector<std::array<float, 2>> faceWeights() {
            // area: 0.5 for reference triangle
            //return { { {1/2.0f, 1.0f} } };  // single point quadrature
            // 4 point quadrature - again weight isn't distributed between points but instead just scales the volume
            // so the first value has the weight for the entire point. 
            return { { {1 / 6.0f, 1.0f},
                {1 / 6.0f, 1.0f},
                {1 / 6.0f, 1.0f} } };
        }
    };
}
