#include "Mesh.hpp"


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

