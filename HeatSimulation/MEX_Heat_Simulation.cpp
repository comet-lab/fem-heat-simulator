/* ========================================================================
* It compiles in MATLAB, but not in Visual Studio... idk.
 *=======================================================================*/

#include "mex.hpp"
#include "mexAdapter.hpp"
#include<string>
#include<memory>
#include<iostream>
#include "FEM_Simulator.h"

 //using namespace matlab::mex;
 //using namespace matlab::data;

class MexFunction : public matlab::mex::Function {
private:
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    FEM_Simulator* simulator;
    std::ostringstream stream;
    bool debug = true;

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
        if (debug) {
            matlab::data::ArrayFactory factory;
            matlabPtr->feval(u"fprintf", 0,
                std::vector<matlab::data::Array>({ factory.createScalar(stream.str()) }));
            // Clear stream buffer
            stream.str("");
        }
    }

    void display3DVector(const std::vector<std::vector<std::vector<float>>>& vec, std::string title) {
        // Get dimensions of the input vector
        size_t dim1 = vec.size();
        size_t dim2 = (dim1 > 0) ? vec[0].size() : 0;
        size_t dim3 = (dim2 > 0) ? vec[0][0].size() : 0;

        stream << "\n" << title << std::endl;
        for (int k = 0; k < dim3; k++) {
            for (int j = 0; j < dim2; j++) {
                for (int i = 0; i < dim1; i++) {
                    stream << simulator->Temp[i][j][k] << ", ";
                    displayOnMATLAB(stream);
                }
                stream << std::endl;
            }
            stream << std::endl;
        }
        displayOnMATLAB(stream);
    }


    /* This is the gateway routine for the MEX-file. */
    void
        operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        try {
            checkArguments(outputs, inputs);
            // Have to convert T0 and NFR to std::vector<<<float>>>
            std::vector<std::vector<std::vector<float>>> T0 = convertMatlabArrayToVector(inputs[0]);
            simulator->setInitialTemperature(T0);
            //display3DVector(simulator->Temp,"Initial Temp: ");

            // Set the NFR
            std::vector<std::vector<std::vector<float>>> NFR = convertMatlabArrayToVector(inputs[1]);
            simulator->setNFR(NFR);

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
            //stream << "Final Time: " << simulator->tFinal << "\nTime step: " << simulator->deltaT << std::endl;
            //displayOnMATLAB(stream);


            // set tissue properties
            float MUA = inputs[5][0];
            float TC = inputs[5][1];
            float VHC = inputs[5][2];
            float HTC = inputs[5][3];
            simulator->setMUA(MUA);
            simulator->setTC(TC);
            simulator->setVHC(VHC);
            simulator->setHTC(HTC);
            //stream << "TC: " << simulator->TC << ", MUA: " << simulator->MUA << ", VHC: " << simulator->VHC << ", HTC: " << simulator->HTC << std::endl;
            //displayOnMATLAB(stream);

            // set boundary conditions
            int boundaryType[6] = { 0,0,0,0,0,0 };
            //stream << "Boundary Conditions: ";
            for (int i = 0; i < 6; i++) {
               boundaryType[i] = inputs[6][i];
            //     stream << boundaryType[0] << ", ";
            }
            stream << std::endl;
            //displayOnMATLAB(stream);
            simulator->setBoundaryConditions(boundaryType);

            // set flux condition
            float Jn = inputs[7][0];
            simulator->setJn(Jn);

            // set ambient temperature
            float ambientTemp = inputs[8][0];
            simulator->setAmbientTemp(ambientTemp);
        }
        catch (const std::exception& e) {
            stream << "Error in Setup: " << std::endl;
            displayOnMATLAB(stream);
            displayError(e.what());
        }

        auto sensorTempsInput = inputs[9];
        std::vector<std::array<float, 3>> sensorTemps;
        for (int s = 0; s < sensorTempsInput.getDimensions()[0]; s++) {
            sensorTemps.push_back({ sensorTempsInput[s][0],sensorTempsInput[s][1] ,sensorTempsInput[s][2] });
        }
        try {
            simulator->setSensorLocations(sensorTemps);
        }
        catch (const std::exception& e){
            displayError(e.what());
        }
        
        Eigen::setNbThreads(Eigen::nbThreads()/2);
        //stream << "Number of threads: " << Eigen::nbThreads() << std::endl;
        //displayOnMATLAB(stream);
        try {
            simulator->createKMF();
        }
        catch (const std::exception& e) {
            stream << "Error in createKMF() " << std::endl;
            displayOnMATLAB(stream);
            displayError(e.what());
        }
        try {
            simulator->performTimeStepping();
        }
        catch (const std::exception& e) {
            stream << "Error in performTimeStepping() " << std::endl;
            displayOnMATLAB(stream);
            displayError(e.what());
        }


        // Have to convert the std::vector to a matlab array for output
        //display3DVector(simulator->Temp, "Final Temp: ");
        matlab::data::TypedArray<float> finalTemp = convertVectorToMatlabArray(simulator->Temp);
        outputs[0] = finalTemp;
        matlab::data::ArrayFactory factory;
        matlab::data::TypedArray<float> sensorTempsOutput = factory.createArray<float>({ simulator->sensorTemps.size(), simulator->sensorTemps[0].size()});
        for (size_t i = 0; i < simulator->sensorTemps.size(); ++i) {
            for (size_t j = 0; j < simulator->sensorTemps[i].size(); ++j) {
                sensorTempsOutput[i][j] = simulator->sensorTemps[i][j];
            }
        }
        outputs[1] = sensorTempsOutput;
    }

    /* This function makes sure that user has provided the proper inputs
    * Inputs: T0, NFR, tissueSize TC, VHC, MUA, HTC, boundaryConditions
     */
    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if (inputs.size() != 10) {
            displayError("Nine inputs required: T0, NFR, tissueSize, tFinal, deltaT tissueProperties, BC, Jn, ambientTemp,sensorLocations");
        }
        if (outputs.size() > 2) {
            displayError("Too many outputs specified.");
        }
        if (outputs.size() < 2) {
            displayError("Not enough outputs specified.");
        }
        if (inputs[0].getType() != matlab::data::ArrayType::SINGLE) {
            displayError("T0 must be an Array of type Single.");
        }
        if (inputs[1].getType() != matlab::data::ArrayType::SINGLE) {
            displayError("NFR must be an Array of type Single.");
        }
        if (!((inputs[1].getDimensions()[0] == inputs[0].getDimensions()[0]) && (inputs[1].getDimensions()[1] == inputs[0].getDimensions()[1])
            && (inputs[1].getDimensions()[2] == inputs[0].getDimensions()[2]))
            &&
            !((inputs[1].getDimensions()[0] == (inputs[0].getDimensions()[0] - 1)) && (inputs[1].getDimensions()[1] == (inputs[0].getDimensions()[1] - 1))
                && (inputs[1].getDimensions()[2] == (inputs[0].getDimensions()[2] - 1)))) {
            displayError("NFR must have either the same dimensions as Temp, or one less in each axis");
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
        if ((inputs[6].getDimensions()[0] != 6) || (inputs[6].getDimensions()[1] != 1)) {
            displayError("Boundary Conditions Size must be 6 x 1");
        }
        if ((inputs[6].getType() != matlab::data::ArrayType::INT32)) {
            displayError("Boundary Conditions must be an int");
        }
        if ((inputs[9].getDimensions()[1] != 3)) {
            displayError("Sensor Locations must be n x 3");
        }
    }
};
