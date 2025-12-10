#pragma once
#include <array>
#include <vector>
#include <Eigen/Dense>
#include <stdexcept>

namespace ShapeFunctions {

    struct TetQuadratic {
        // 10-node quadratic tetrahedron
        // Node ordering (vertices first, then midside edges):
        // 0,1,2,3 = vertices
        // 4 = edge 0-1, 5 = edge 1-2, 6 = edge 2-0, 7 = edge 0-3, 8 = edge 1-3, 9 = edge 2-3

        static constexpr int nNodes = 10;       // 10 point serendipity element
        static constexpr int nGP = 11;          // 11-point volume Gauss points from Keast 1985 "MODERATE-DEGREE TETRAHEDRAL QUADRATURE FORMULAS"
        static constexpr int nFaceGP = 6;      // 6-point triangular face quadrature from Dunavant 
        static constexpr int nFaceNodes = 6;   // 6-node quadratic triangle
        static constexpr int nFaces = 4;

        // Face connectivity (6 nodes per triangular face)
        // Order: [vertex0, vertex1, vertex2, mid01, mid12, mid20]
        static constexpr std::array<std::array<int, 6>, 4> faceConnectivity = { {
            {1, 2, 3, 5, 9, 8},  // face opposite node 0
            {0, 3, 2, 7, 9, 6},  // face opposite node 1
            {0, 1, 3, 4, 8, 7},  // face opposite node 2
            {0, 2, 1, 6, 5, 4}   // face opposite node 3
        } };

        // Quadratic tetrahedral shape functions in barycentric coords
        static inline float N(const std::array<float, 3>& xi, int A) {
            float l0 = 1.0f - xi[0] - xi[1] - xi[2];
            float l1 = xi[0];
            float l2 = xi[1];
            float l3 = xi[2];
            switch (A) {
            case 0: return l0 * (2 * l0 - 1);
            case 1: return l1 * (2 * l1 - 1);
            case 2: return l2 * (2 * l2 - 1);
            case 3: return l3 * (2 * l3 - 1);
            case 4: return 4 * l0 * l1;
            case 5: return 4 * l1 * l2;
            case 6: return 4 * l2 * l0;
            case 7: return 4 * l0 * l3;
            case 8: return 4 * l1 * l3;
            case 9: return 4 * l2 * l3;
            default: return 0.0f;
            }
        }

        // Derivatives wrt xi,eta,zeta
        static inline Eigen::RowVector3f dNdxi(const std::array<float, 3>& xi, int A) {
            float l0 = 1.0f - xi[0] - xi[1] - xi[2];
            float l1 = xi[0];
            float l2 = xi[1];
            float l3 = xi[2];
            switch (A) {
            case 0: return Eigen::RowVector3f(- 4 * l0 + 1, - 4 * l0 + 1, -4 * l0 + 1); // d [l0 * (2 * l0 - 1)] / dxi
            case 1: return Eigen::RowVector3f(4 * l1 - 1, 0.0f, 0.0f);
            case 2: return Eigen::RowVector3f(0.0f, 4 * l2 - 1, 0.0f);
            case 3: return Eigen::RowVector3f(0.0f, 0.0f, 4 * l3 - 1);
            case 4: return Eigen::RowVector3f(4 * (l0 - l1), -4 * l1, -4 * l1);
            case 5: return Eigen::RowVector3f(4 * l2, 4 * l1, 0.0f);
            case 6: return Eigen::RowVector3f(-4 * l2, 4 * (l0 - l2), -4 * l2);
            case 7: return Eigen::RowVector3f(-4 * l3, -4 * l3, 4 * (l0 - l3));
            case 8: return Eigen::RowVector3f(4 * l3, 0.0f, 4 * l1);
            case 9: return Eigen::RowVector3f(0.0f, 4 * l3, 4 * l2);
            default: return Eigen::RowVector3f::Zero();
            }
        }

        // Face shape function (6-node triangle)
        static inline float N_face(const std::array<float, 2>& xi, int A) {
            float r = xi[0], s = xi[1];
            float t = 1.0f - r - s;
            switch (A) {
            case 0: return t * (2 * t - 1);
            case 1: return r * (2 * r - 1);
            case 2: return s * (2 * s - 1);
            case 3: return 4 * t * r;
            case 4: return 4 * r * s;
            case 5: return 4 * s * t;
            default: return 0.0f;
            }
        }

        // Face derivatives wrt xi, eta
        static inline Eigen::RowVector2f dNdxi_face(const std::array<float, 2>& xi, int A) {
            float r = xi[0], s = xi[1];
            float t = 1.0f - r - s;
            switch (A) {
            case 0: return Eigen::RowVector2f(-4 * t + 1, -4 * t + 1);
            case 1: return Eigen::RowVector2f(4 * r - 1, 0.0f);
            case 2: return Eigen::RowVector2f(0.0f, 4 * s - 1);
            case 3: return Eigen::RowVector2f(4 * (t - r), -4 * r);
            case 4: return Eigen::RowVector2f(4 * s, 4 * r);
            case 5: return Eigen::RowVector2f(-4 * s, 4 * (t - s));
            default: return Eigen::RowVector2f::Zero();
            }
        }

        // Map face Gauss point to tetrahedral coordinates
        static inline std::array<float, 3> mapFaceGPtoXi(const std::array<float, 2>& gp, int face) {
            const float r = gp[0], s = gp[1];
            std::array<float, 3> xi{};
            switch (face) {
            case 0: xi = { 1 - r - s, r, s }; break;     // nodes 1,2,3 opposite node 0
            case 1: xi = { 0.0f, s, r }; break;      // nodes 0,3,2 opposite node 1
            case 2: xi = { r, 0.0f, s }; break;      // nodes 0,1,3 opposite node 2
            case 3: xi = { s, r, 0.0f }; break;      // nodes 0,2,1 opposite node 3
            default: throw std::runtime_error("Invalid face index in TetQuadratic::mapFaceGPtoXi");
            }
            return xi;
        }

        // Volume Gauss points (example 5-point)
        static inline std::vector<std::array<float, 3>> gaussPoints() {
            float alpha = 1 / 4.0f;
            float beta = 0.78571428571429f;
            float gamma = 0.07142857142857f;
            float delta = 0.3994035762f;
            float rho = 0.1005964238f;
            return { {{alpha, alpha, alpha}}, // centroid
                {{beta, gamma, gamma}},
                {{gamma, beta, gamma}},
                {{gamma, gamma, beta}},
                {{gamma, gamma, gamma}},
                {{delta, rho, rho}},
                {{rho, delta, rho}},
                {{rho, rho, delta}},
                {{rho, delta, delta}},
                {{delta, rho, delta}},
                {{delta, delta, rho}} };
        }

        static inline std::vector<std::array<float, 3>> weights() {
            float w1 = -0.0131555555555556f;
            float w2 = 0.007622222222222f;
            float w3 = 0.024888888888889f;
            return { {{w1,1.0f,1.0f}},
                {{w2,1.0f,1.0f}},
                {{w2,1.0f,1.0f}},
                {{w2,1.0f,1.0f}},
                {{w2,1.0f,1.0f}},
                {{w3,1.0f,1.0f}},
                {{w3,1.0f,1.0f}},
                {{w3,1.0f,1.0f}},
                {{w3,1.0f,1.0f}},
                {{w3,1.0f,1.0f}},
                {{w3,1.0f,1.0f}} };
        }

        // Face Gauss points (quadratic triangle)
        static inline std::vector<std::array<float, 2>> faceGaussPoints() {
            float alpha = 0.091576213509771f;
            float beta = 0.816847572980458f;
            float gamma = 0.445948490915965f;
            float delta = 0.108103018168070f;
            return { {{alpha, alpha}},
                {{alpha, beta}},
                {{beta, alpha}},
                {{gamma, gamma}},
                {{gamma, delta}},
                {{delta, gamma}} };
        }

        static inline std::vector<std::array<float, 2>> faceWeights() {
            float w1 = 0.054975871827661;
            float w2 = 0.111690794839005f;
            return { {{w1, 1.0f}},
                {{w1, 1.0f}},
                {{w1, 1.0f}},
                {{w2, 1.0f}},
                {{w2, 1.0f}}, 
                {{w2, 1.0f}} };
        }
    };
}
