#include "Mesh.hpp"


// =============================
// FACE CONNECTIVITY DEFINITIONS
// =============================
const std::array<std::array<int, 4>, 6> FaceConnectivity::HEX8 = { {
	{0, 1, 3, 2},
	{4, 5, 7, 6},
	{0, 1, 5, 4},
	{2, 3, 7, 6},
	{0, 2, 6, 4},
	{1, 3, 7, 5}
} };

const std::array<std::array<int, 8>, 6> FaceConnectivity::HEX20 = { {
	{0, 1, 3, 2, 8, 9, 11, 10},
	{4, 5, 7, 6, 12, 13, 15, 14},
	{0, 1, 5, 4, 8, 17, 12, 16},
	{2, 3, 7, 6, 10, 19, 14, 18},
	{0, 2, 6, 4, 16, 18, 14, 12},
	{1, 3, 7, 5, 17, 19, 15, 13}
} };

const std::array<std::array<int, 3>, 4> FaceConnectivity::TET4 = { {
	{0, 2, 1},
	{0, 1, 3},
	{1, 2, 3},
	{2, 0, 3}
} };

const std::array<std::array<int, 6>, 4> FaceConnectivity::TET10 = { {
	{0, 2, 1, 6, 5, 4},
	{0, 1, 3, 4, 8, 7},
	{1, 2, 3, 5, 9, 8},
	{2, 0, 3, 6, 7, 9}
} };


Mesh::Mesh()
{
}

Mesh::Mesh(std::vector<Node> nodes, std::vector<Element> elements, std::vector<BoundaryFace> boundaryFaces)
{
	setNodes(nodes);
	setElements(elements);
	setBoundaryFaces(boundaryFaces);

}

void Mesh::setNodes(std::vector<Node> nodes)
{
	nodes_ = nodes;
}

void Mesh::setBoundaryFaces(std::vector<BoundaryFace> boundaryFaces)
{
	boundaryFaces_ = boundaryFaces;
}

void Mesh::setElements(std::vector<Element> elements)
{
	elements_ = elements;
	setOrderAndShape();
}

void Mesh::setOrderAndShape()
{
	int nNodesPerElem = elements_[0].nodes.size();
	switch (nNodesPerElem)
	{
	case 4:
		order_ = GeometricOrder::LINEAR;
		elementShape_ = TETRAHEDRAL;
		break;
	case 8:
		order_ = LINEAR;
		elementShape_ = HEXAHEDRAL;
		break;
	case 10:
		order_ = QUADRATIC;
		elementShape_ = TETRAHEDRAL;
		break;
	case 20:
		order_ = QUADRATIC;
		elementShape_ = HEXAHEDRAL;
		break;
	default:
		throw std::runtime_error("Invalid number of nodes per element");
	}

	for (int e = 0; e < elements_.size(); e++)
	{
		if (elements_[e].nodes.size() != nNodesPerElem)
		{
			throw std::runtime_error("Each element should have the same number of nodes");
		}
	}
}