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
    std::ostringstream stream;

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

    /* Helper function to generate messages to matlab command window
    */
    void displayOnMATLAB(std::ostringstream& stream) {
        // Pass stream content to MATLAB fprintf function
        matlab::data::ArrayFactory factory;
        matlabPtr->feval(u"fprintf", 0,
            std::vector<matlab::data::Array>({ factory.createScalar(stream.str()) }));
        // Clear stream buffer
        stream.str("");
    }


    /* This is the gateway routine for the MEX-file. */
    void
        operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {

        checkArguments(outputs, inputs);
        // Have to convert T0 and NFR to std::vector<<<float>>>
        std::vector<std::vector<std::vector<float>>> T0 = convertMatlabArrayToVector(inputs[0]);
        simulator->setInitialTemperature(T0);
        stream << "Initial Temperature: " << std::endl;
        for (int k = 0; k < 3; k++) {
            for (int j = 0; j < 3; j++) {
                for (int i = 0; i < 3; i++) {
                    stream << simulator->Temp[i][j][k] << ", ";
                    displayOnMATLAB(stream);
                }
                stream << std::endl;
            }
            stream << std::endl;
        }
        stream << std::endl;
        displayOnMATLAB(stream);



        // Set tissue size
        float tissueSize[3];
        tissueSize[0] = inputs[2][0];
        tissueSize[1] = inputs[2][1];
        tissueSize[2] = inputs[2][2];
        simulator->setTissueSize(tissueSize);

        // set time step and final time
        float tFinal = inputs[3][0];
        float deltaT = inputs[4][0];
        simulator->tFinal = tFinal;
        simulator->deltaT = deltaT;

        // set tissue properties
        float MUA = inputs[5][0];
        float TC = inputs[5][1];
        float VHC = inputs[5][2];
        float HTC = inputs[5][3];
        simulator->setTC(TC);
        simulator->setVHC(VHC);
        simulator->setMUA(MUA);
        simulator->setHTC(HTC);

        // set boundary conditions
        int boundaryType[6] = { 0,0,0,0,0,0 };
        for (int i = 0; i < 6; i++) {
            boundaryType[i] = inputs[6][i];
            stream << boundaryType[0] << ", ";
        }
        stream << std::endl;
        displayOnMATLAB(stream);
        simulator->setBoundaryConditions(boundaryType);
        
        // set flux condition
        float Jn = inputs[7][0];
        simulator->Jn = Jn;

        // set ambient temperature
        float ambientTemp = inputs[8][0];
        simulator->setAmbientTemp(ambientTemp);

        // Run the FEA
        std::vector<std::vector<std::vector<float>>> NFR = convertMatlabArrayToVector(inputs[1]);
        simulator->solveFEA(NFR);

        // Have to convert the std::vector to a matlab array for output
        stream << "Final Temperature: " << std::endl;
        matlab::data::TypedArray<float> finalTemp = convertVectorToMatlabArray(simulator->Temp);
        for (int k = 0; k < 3; k++) {
            for (int j = 0; j < 3; j++) {
                for (int i = 0; i < 3; i++) {
                    stream << simulator->Temp[i][j][k] << ", ";
                    displayOnMATLAB(stream);
                }
                stream << std::endl;
            }
            stream << std::endl;
        }
        stream << std::endl;
        displayOnMATLAB(stream);
        outputs[0] = finalTemp;
    }

    /* This function makes sure that user has provided the proper inputs
    * Inputs: T0, NFR, tissueSize TC, VHC, MUA, HTC, boundaryConditions
     */
    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if (inputs.size() != 9) {
            displayError("Nine inputs required: T0, NFR, tissueSize, tFinal, deltaT tissueProperties, BC, Jn, ambientTemp");
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
        /* This Check Doesnt work
        if (inputs[4][0] > inputs[3][0]) {
            displayError("deltaT must be less than the final time");
        }
        */
        if ((inputs[5].getDimensions()[0] != 4) || (inputs[5].getDimensions()[1] != 1)) {
            displayError("Tissue Properties must be 4 x 1: MUA, TC, VHC, HTC");
        }
        if ((inputs[6].getDimensions()[0] != 6) || (inputs[7].getDimensions()[1] != 1)) {
            displayError("Boundary Conditions Size must be 6 x 1");
        }
        if ((inputs[6].getType() != matlab::data::ArrayType::INT32)) {
            displayError("Boundary Conditions must be an int");
        }
    }
};
