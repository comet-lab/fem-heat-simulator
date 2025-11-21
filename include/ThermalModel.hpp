#pragma once
#include "Eigen/Dense"

struct ThermalModel
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	float TC = 0; // Thermal Conductivity [W/cm C]
	float VHC = 0; // Volumetric Heat Capacity [W/cm^3]
	float MUA = 0; // Absorption Coefficient [cm^-1]
	float HTC = 0; // convective heat transfer coefficient [W/cm^2]
	float ambientTemp = 0;  // Temperature surrounding the tissue for Convection [C]
	Eigen::VectorXf Temp; // Our values for temperature at the nodes of the elements
	Eigen::VectorXf fluenceRate; // Our values for Heat addition
	Eigen::VectorXf fluenceRateElem; // Our values for Heat addition
	float heatFlux = 0; // heat escaping the Neumann Boundary
};