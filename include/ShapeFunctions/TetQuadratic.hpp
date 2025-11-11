#pragma once
#include <Eigen/Dense>
#include <stdexcept>

namespace ShapeFunctions {

    /// Quadratic 10-node tetrahedral element
    struct TetQuadratic
    {
        static constexpr int NumNodes = 10;

        /// Shape function value N_a(xi) at local coordinates xi
        static float value(int a, const Eigen::Vector3f& xi);

        /// Gradient of shape function dN_a/dxi
        static Eigen::Vector3f gradient(int a, const Eigen::Vector3f& xi);
    };

} // namespace ShapeFunctions
