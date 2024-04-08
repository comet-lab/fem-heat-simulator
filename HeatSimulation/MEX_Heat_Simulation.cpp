/* ========================================================================
* It compiles in MATLAB, but not in Visual Studio... idk.
 *=======================================================================*/

#include "mex.hpp"
#include "mexAdapter.hpp"
#include<string>
#include<memory>
#include<iostream>
#include "FEM_Simulator.h"
#include "FEM_Simulator.cpp"

 //using namespace matlab::mex;
 //using namespace matlab::data;

class MexFunction : public matlab::mex::Function {
private:
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    FEM_Simulator* simulator;
public:
    /* Constructor for the class. */
    MexFunction()
    {
        matlabPtr = getEngine();
        simulator = new FEM_Simulator();
    }
    /* Helper function to convert a matlab array to a std vector*/
    std::vector<std::vector<std::vector<float>>> convertMatlabArrayToVector(const matlab::data::Array& matlabArray) {
        std::vector<std::vector<std::vector<float>>> result;

        // Get dimensions of the Matlab array
        size_t dim1 = matlabArray.getDimensions()[0];
        size_t dim2 = matlabArray.getDimensions()[1];
        size_t dim3 = matlabArray.getDimensions()[2];

        // Resize the result vector to match the dimensions of the Matlab array
        result.resize(dim1);
        for (size_t i = 0; i < dim1; ++i) {
            result[i].resize(dim2);
            for (size_t j = 0; j < dim2; ++j) {
                result[i][j].resize(dim3);
            }
        }

        // Iterate through the Matlab array and fill the result vector
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    result[i][j][k] = static_cast<float>(matlabArray[i][j][k]);
                }
            }
        }

        return result;
    }

    matlab::data::TypedArray<float> convertVectorToMatlabArray(const std::vector<std::vector<std::vector<float>>>& vec) {
        // Get dimensions of the input vector
        size_t dim1 = vec.size();
        size_t dim2 = (dim1 > 0) ? vec[0].size() : 0;
        size_t dim3 = (dim2 > 0) ? vec[0][0].size() : 0;

        matlab::data::ArrayFactory factory;
        // Create MATLAB array with appropriate dimensions
        matlab::data::TypedArray<float> matlabArray = factory.createArray<float>({ dim1, dim2, dim3 });

        // Fill MATLAB array with vector data
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    matlabArray[i][j][k] = vec[i][j][k];
                }
            }
        }

        return matlabArray;
    }

    /* Helper function to generate an error message from given string,
     * and display it over MATLAB command prompt.
     */
    void displayError(std::string errorMessage)
    {
        matlab::data::ArrayFactory factory;
        matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({
          factory.createScalar(errorMessage) }));
    }


    /* This is the gateway routine for the MEX-file. */
    void
        operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {

        checkArguments(outputs, inputs);
        // Have to convert T0 and NFR to std::vector<<<float>>>
        std::vector<std::vector<std::vector<float>>> T0 = convertMatlabArrayToVector(inputs[0]);
        simulator->setInitialTemperature(T0);
        float tissueSize[3];
        tissueSize[0] = inputs[2][0];
        tissueSize[1] = inputs[2][1];
        tissueSize[2] = inputs[2][2];
        simulator->setTissueSize(tissueSize);
        float TC = inputs[3][0];
        float VHC = inputs[4][0];
        float MUA = inputs[5][0];
        float HTC = inputs[6][0];
        simulator->setTC(TC);
        simulator->setVHC(VHC);
        simulator->setMUA(MUA);
        simulator->setHTC(HTC);

        int boundaryType[6] = { 0,0,0,0,0,0 };
        for (int i = 0; i < 6; i++) {
            boundaryType[i] = inputs[7][i];
        }
        simulator->setBoundaryConditions(boundaryType);

        std::vector<std::vector<std::vector<float>>> NFR = convertMatlabArrayToVector(inputs[1]);
        simulator->solveFEA(NFR);
        // Have to convert the std::vector to a matlab array
        outputs[0] = convertVectorToMatlabArray(simulator->Temp);
    }

    /* This function makes sure that user has provided the proper inputs
    * Inputs: T0, NFR, tissueSize TC, VHC, MUA, HTC, boundaryConditions
     */
    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if (inputs.size() != 8) {
            displayError("Seven inputs required: T0, NFR, tissueSize TC, VHC, MUA, HTC, BC");
        }
        if (outputs.size() > 1) {
            displayError("Too many outputs specified.");
        }
        if (inputs[0].getType() != matlab::data::ArrayType::SINGLE) {
            displayError("T0 must be an Array of type Single.");
        }
        if (inputs[1].getType() != matlab::data::ArrayType::SINGLE) {
            displayError("NFR must be an Array of type Single.");
        }
        if ((inputs[2].getDimensions()[0] != 3) || (inputs[2].getDimensions()[1] != 1)) {
            displayError("Tissue Size must be 3 x 1");
        }
        if ((inputs[7].getDimensions()[0] != 6) || (inputs[7].getDimensions()[1] != 1)) {
            displayError("Boundary Conditions Size must be 6 x 1");
        }
        if ((inputs[7].getType() != matlab::data::ArrayType::INT32)) {
            displayError("Boundary Conditions must be an int");
        }
    }
};
