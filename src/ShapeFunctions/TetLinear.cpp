#include "ShapeFunctions/TetLinear.hpp"

namespace ShapeFunctions {

    // Shape function N_a(xi)
    float TetLinear::value(int a, const Eigen::Vector3f& xi)
    {
        switch (a)
        {
        case 0: return 1.f - xi[0] - xi[1] - xi[2];
        case 1: return xi[0];
        case 2: return xi[1];
        case 3: return xi[2];
        default: throw std::runtime_error("TetLinear::value: invalid node index");
        }
    }

    // Gradient dN_a/dxi
    Eigen::Vector3f TetLinear::gradient(int a)
    {
        switch (a)
        {
        case 0: return Eigen::Vector3f(-1.f, -1.f, -1.f);
        case 1: return Eigen::Vector3f(1.f, 0.f, 0.f);
        case 2: return Eigen::Vector3f(0.f, 1.f, 0.f);
        case 3: return Eigen::Vector3f(0.f, 0.f, 1.f);
        default: throw std::runtime_error("TetLinear::gradient: invalid node index");
        }
    }

} // namespace ShapeFunctions
