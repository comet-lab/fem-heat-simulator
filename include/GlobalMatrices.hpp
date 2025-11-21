#pragma once
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <vector>

struct GlobalMatrices
{
	std::vector<long> nodeMap;
	long nNonDirichlet = 0;
	std::vector<long> validNodes;
	Eigen::SparseMatrix<float> M; // Thermal Mass Matrix
	Eigen::SparseMatrix<float> K; // Thermal Conductivity Matrix
	Eigen::SparseMatrix<float> Q; // Convection Matrix -- should have the same structure as Me just gets scaled by htc instead of vhc
	// Internal nodal heat generation (aka laser). Its size is nNodes x nNodes. Becomes a vector once post multiplied by a 
	// vector dictating the fluence experienced at each node. 
	Eigen::SparseMatrix<float> Fint;
	// forcing function due to irradiance when using elemental fluence rate. Its size is nNodes x nElems. Becomes vector once post multiplied
	// by a vector dictating the fluence experienced by each element
	Eigen::SparseMatrix<float> FintElem;
	// forcing function due to convection on dirichlet node. Size is nNodes x nNodes. Becomes a vector once post multiplied
	// by a vector specifying the fixed temperature at each element.
	Eigen::SparseMatrix<float> Fconv;
	// Forcing Function due to conductivity matrix on dirichlet nodes. Stored as a matrix but becomes vector once multiplied by nodal temperatures
	Eigen::SparseMatrix<float> Fk;
	Eigen::VectorXf Fflux; // forcing function due to constant heatFlux boundary
	Eigen::VectorXf Fq; // forcing function due to ambient temperature
};