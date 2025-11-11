#pragma once
#include <Eigen/Dense>
#include <stdexcept>

namespace ShapeFunctions {

    /// Linear 4-node tetrahedral element
    struct TetLinear
    {
        static constexpr int NumNodes = 4;

        /// Shape function value N_a(xi) at local coordinates xi
        /// xi: (xi, eta, zeta) in reference tetrahedron
        static float value(int a, const Eigen::Vector3f& xi);

        /// Gradient of shape function dN_a/dxi (constant for linear tet)
        static Eigen::Vector3f gradient(int a);
    };

} // namespace ShapeFunctions
