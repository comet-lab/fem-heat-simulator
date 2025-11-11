#include "ShapeFunctions/TetQuadratic.hpp"

namespace ShapeFunctions {

    // Shape function value N_a(xi)
    float TetQuadratic::value(int a, const Eigen::Vector3f& xi)
    {
        float L2 = xi[0];
        float L3 = xi[1];
        float L4 = xi[2];
        float L1 = 1.f - L2 - L3 - L4;

        switch (a)
        {
        case 0: return L1 * (2.f * L1 - 1.f);
        case 1: return L2 * (2.f * L2 - 1.f);
        case 2: return L3 * (2.f * L3 - 1.f);
        case 3: return L4 * (2.f * L4 - 1.f);

        case 4: return 4.f * L1 * L2;
        case 5: return 4.f * L2 * L3;
        case 6: return 4.f * L3 * L1;
        case 7: return 4.f * L1 * L4;
        case 8: return 4.f * L2 * L4;
        case 9: return 4.f * L3 * L4;

        default:
            throw std::runtime_error("TetQuadratic::value: invalid node index");
        }
    }

    // Gradient dN_a/dxi
    Eigen::Vector3f TetQuadratic::gradient(int a, const Eigen::Vector3f& xi)
    {
        float L2 = xi[0];
        float L3 = xi[1];
        float L4 = xi[2];
        float L1 = 1.f - L2 - L3 - L4;

        // gradients of linear barycentric coordinates
        Eigen::Vector3f gL1(-1.f, -1.f, -1.f);
        Eigen::Vector3f gL2(1.f, 0.f, 0.f);
        Eigen::Vector3f gL3(0.f, 1.f, 0.f);
        Eigen::Vector3f gL4(0.f, 0.f, 1.f);

        switch (a)
        {
            // corner nodes
        case 0: return gL1 * (4.f * L1 - 1.f);
        case 1: return gL2 * (4.f * L2 - 1.f);
        case 2: return gL3 * (4.f * L3 - 1.f);
        case 3: return gL4 * (4.f * L4 - 1.f);

            // edge nodes
        case 4: return 4.f * (L1 * gL2 + L2 * gL1);
        case 5: return 4.f * (L2 * gL3 + L3 * gL2);
        case 6: return 4.f * (L3 * gL1 + L1 * gL3);
        case 7: return 4.f * (L1 * gL4 + L4 * gL1);
        case 8: return 4.f * (L2 * gL4 + L4 * gL2);
        case 9: return 4.f * (L3 * gL4 + L4 * gL3);

        default:
            throw std::runtime_error("TetQuadratic::gradient: invalid node index");
        }
    }

} // namespace ShapeFunctions
