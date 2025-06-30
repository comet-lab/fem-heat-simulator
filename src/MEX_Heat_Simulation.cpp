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
    bool silentMode = true;
    bool useAllCPUs = true;
    bool createAllMatrices = true;
    bool createFirrMatrix = true;
    float layerHeight = 1;
    int elemsInLayer = 1;
    int Nn1d = 2;

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
        if (!this->silentMode) {
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
                    stream << vec[i][j][k] << ", ";
                    displayOnMATLAB(stream);
                }
                stream << std::endl;
            }
            stream << std::endl;
        }
        displayOnMATLAB(stream);
    }


    /* This is the gateway routine for the MEX-file. */
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        auto startTime = std::chrono::high_resolution_clock::now();
        checkArguments(outputs, inputs);
        stream.str("");
        stream << "Start of () function" << std::endl;
        displayOnMATLAB(stream);
        try {
            
            // Have to convert T0 and FluenceRate to std::vector<<<float>>>
            std::vector<std::vector<std::vector<float>>> T0 = convertMatlabArrayToVector(inputs[0]);
            std::vector<std::vector<std::vector<float>>> FluenceRate = convertMatlabArrayToVector(inputs[1]);

            // Get tissue size
            float tissueSize[3];
            tissueSize[0] = inputs[2][0];
            tissueSize[1] = inputs[2][1];
            tissueSize[2] = inputs[2][2];
            
            // Get time step and final time
            float tFinal = inputs[3][0];
            float deltaT = inputs[4][0];

            // Extract tissue properties
            float MUA = inputs[5][0];
            float TC = inputs[5][1];
            float VHC = inputs[5][2];
            float HTC = inputs[5][3];

            // Get boundary conditions
            int boundaryType[6] = { 0,0,0,0,0,0 };
            stream << "Boundary Conditions: ";
            for (int i = 0; i < 6; i++) {
               boundaryType[i] = inputs[6][i];
               stream << boundaryType[i] << ", ";
            }
            stream << std::endl;
            displayOnMATLAB(stream);
            

            // get heatFlux condition
            float heatFlux = inputs[7][0];
            // get ambient temperature
            float ambientTemp = inputs[8][0];

            /* SET ALL PARAMETERS NECESSARY BEFORE CONSTRUCTING ELEMENTS*/
            // First check to see if certain variables have changed which would require us to reconstruct elements
            if (!this->createAllMatrices) {
                this->createAllMatrices = checkForMatrixReset(Nn1d, T0, FluenceRate, tissueSize, layerHeight, elemsInLayer, boundaryType);
            }
            // Set the type of basis functions we are using by setting nodes per dimension of an 
            this->simulator->Nn1d = Nn1d;
            this->simulator->setTemp(T0);
            // Set the FluenceRate
            this->simulator->setFluenceRate(FluenceRate);
            this->simulator->setTissueSize(tissueSize);
            // set the layer info
            this->simulator->setLayer(layerHeight, elemsInLayer);
            // set the final time
            this->simulator->tFinal = tFinal;
            // set the time step
            this->simulator->deltaT = deltaT;
            // set the tissue properties
            this->simulator->setMUA(MUA);
            this->simulator->setTC(TC);
            this->simulator->setVHC(VHC);
            this->simulator->setHTC(HTC);
            // set boundary conditions
            this->simulator->setBoundaryConditions(boundaryType);
            // set heatFlux
            this->simulator->setFlux(heatFlux);

            //print statements 
            stream << "Final Time: " << this->simulator->tFinal << "\nTime step: " << this->simulator->deltaT << std::endl;
            displayOnMATLAB(stream);
            stream << "TC: " << this->simulator->TC << ", MUA: " << this->simulator->MUA << ", VHC: " << this->simulator->VHC << ", HTC: " << this->simulator->HTC << std::endl;
            displayOnMATLAB(stream);

            this->simulator->setAmbientTemp(ambientTemp);
        }
        catch (const std::exception& e) {
            stream << "Error in Setup: " << std::endl;
            displayOnMATLAB(stream);
            displayError(e.what());
            return;
        }
        
        // Set sensor locations
        auto sensorTempsInput = inputs[9];
        std::vector<std::array<float, 3>> sensorTemps;
        for (int s = 0; s < sensorTempsInput.getDimensions()[0]; s++) {
            sensorTemps.push_back({ sensorTempsInput[s][0],sensorTempsInput[s][1] ,sensorTempsInput[s][2] });
        }
        try {
            this->simulator->setSensorLocations(sensorTemps);
        }
        catch (const std::exception& e){
            stream << "Error setting sensor locations " << std::endl;
            displayOnMATLAB(stream);
            displayError(e.what());
            return;
        }
        
        // Set parallelization
        Eigen::setNbThreads(1);
#ifdef _OPENMP
        stream << "OPEMMP Enabled" << std::endl;
        displayOnMATLAB(stream);
        if (useAllCPUs) { //useAllCPUs is true
            Eigen::setNbThreads(omp_get_num_procs()/2);
        }
#endif
        stream << "Number of threads: " << Eigen::nbThreads() << std::endl;
        displayOnMATLAB(stream);

        // Create global K M and F 
        if (this->createAllMatrices) { // only need to create the KMF matrices the first time
            this->createAllMatrices = false;
            this->createFirrMatrix = false;
            try {
                this->simulator->createKMF();
                stream << "Global matrices created" << std::endl;
                displayOnMATLAB(stream);
            }
            catch (const std::exception& e) {
                stream << "Error in createKMF() " << std::endl;
                displayOnMATLAB(stream);
                displayError(e.what());
                return;
            }
        }
        else if (this->createFirrMatrix) {
            this->createFirrMatrix = false;
            this->simulator->createFirr();
            stream << "Firr Matrix created" << std::endl;
            displayOnMATLAB(stream);
        }

        // Perform time stepping
        try { //
            this->simulator->performTimeStepping();
            stream << "Time Stepping Complete" << std::endl;
            displayOnMATLAB(stream);
        }
        catch (const std::exception& e) {
            stream << "Error in performTimeStepping() " << std::endl;
            displayOnMATLAB(stream);
            displayError(e.what());
            return;
        }


        // Have to convert the std::vector to a matlab array for output
        //display3DVector(this->simulator->Temp, "Final Temp: ");
        std::vector<std::vector<std::vector<float>>> TFinal = this->simulator->getTemp();
        matlab::data::TypedArray<float> finalTemp = convertVectorToMatlabArray(TFinal);
        outputs[0] = finalTemp;
        matlab::data::ArrayFactory factory;
        matlab::data::TypedArray<float> sensorTempsOutput = factory.createArray<float>({ this->simulator->sensorTemps.size(), this->simulator->sensorTemps[0].size()});
        for (size_t i = 0; i < this->simulator->sensorTemps.size(); ++i) {
            for (size_t j = 0; j < this->simulator->sensorTemps[i].size(); ++j) {
                sensorTempsOutput[i][j] = this->simulator->sensorTemps[i][j];
            }
        }
        outputs[1] = sensorTempsOutput;

        auto stopTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
        stream << "End of MEX() Function:  " << duration.count() / 1000000.0 << std::endl;
        displayOnMATLAB(stream);
    }

    /* This function makes sure that user has provided the proper inputs
    * Inputs: T0, FluenceRate, tissueSize TC, VHC, MUA, HTC, boundaryConditions
     */
    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if (inputs.size() < 10) {
            displayError("At least 10 inputs required: T0, NFR, tissueSize, tFinal,"
                "deltaT tissueProperties, BC, Jn, ambientTemp, sensorLocations, (useAllCPUs), (silentMode), (layers), (Nn1d), (createAllMatrices) ");
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
        if (inputs.size() > 10) {
            useAllCPUs = inputs[10][0];
        }
        if (inputs.size() > 11) {
            this->silentMode = inputs[11][0];
            this->simulator->silentMode = this->silentMode;
        }
        if (inputs.size() > 12) {
            layerHeight = inputs[12][0];
            elemsInLayer = int(inputs[12][1]);
        }
        else {
            elemsInLayer = inputs[0].getDimensions()[2];
            layerHeight = inputs[2][2];
        }
        if (inputs.size() > 13) {
            Nn1d = inputs[13][0];
        }
        if (inputs.size() > 14) { 
            // this parameter is primarily here for debugging and timing tests. Not really practical for someone
            // to use this while they're running simulations
            this->createAllMatrices = this->createAllMatrices || inputs[14][0];
        }
    }

    bool checkForMatrixReset(int Nn1d, std::vector<std::vector<std::vector<float>>>& T0, std::vector<std::vector<std::vector<float>>>& FluenceRate,
        float tissueSize[3],float layerHeight, int elemsInLayer,int boundaryType[6]) {
        // this function checks to see if we need to recreate the global element matrices
        this->createAllMatrices = false;

        // if the number of nodes in 1 dimension for an element changes, we need to reconstruct the matrices
        if (this->simulator->Nn1d != Nn1d) {
            this->createAllMatrices = true;
        }
        // if the size of our temperature vector changes, we need to reconstruct the matrices
        if (T0.size() != this->simulator->nodesPerAxis[0]) this->createAllMatrices = true; 
        if (T0[0].size() != this->simulator->nodesPerAxis[1]) this->createAllMatrices = true;
        if (T0[0][0].size() != this->simulator->nodesPerAxis[2]) this->createAllMatrices = true;

        // if our FluenceRate has changed, we need to reconstruct at least FirrElem
        // TODO make it so that we only reconstruct FirrElem instead of all of them
        for (int k = 0; k < FluenceRate[0][0].size(); k++) {
            if (this->createFirrMatrix || this->createAllMatrices) break; // flag for breaking out of nested loop
            for (int j = 0; j < FluenceRate[0].size(); j++) {
                if (this->createFirrMatrix) break; // flags for breaking out of nested loop
                for (int i = 0; i < FluenceRate.size(); i++) {
                    if (abs(this->simulator->FluenceRate(i + j*FluenceRate.size() + k*FluenceRate.size()*FluenceRate[0].size()) - FluenceRate[i][j][k]) > 0.0001) { // check if difference is greater than 1e-4
                        this->createFirrMatrix = true;
                        break;
                    }
                }
            }
        }

        // if our tissue size has changed we need to reconstruct all matrices because our jacobian has changed
        if (tissueSize[0] != this->simulator->tissueSize[0]) this->createAllMatrices = true;
        if (tissueSize[1] != this->simulator->tissueSize[1]) this->createAllMatrices = true;
        if (tissueSize[2] != this->simulator->tissueSize[2]) this->createAllMatrices = true;

        //if layer height or layer size changed then again we have a jacobian change
        if (abs(layerHeight - this->simulator->layerHeight) > 0.0001) this->createAllMatrices = true;
        if (elemsInLayer != elemsInLayer) this->createAllMatrices = true;

        // check if any boundaries have changed
        for (int b = 0; b < 6; b++) {
            if (boundaryType[b] != this->simulator->boundaryType[b]) {
                this->createAllMatrices = true;
                break;
            }
        }

        if (this->createAllMatrices) {
            stream << "Need to recreate Matrices" << std::endl;
            displayOnMATLAB(stream);
        }
        else if (this->createFirrMatrix) {
            stream << "Need to recreate Firr Matrix Only" << std::endl;
            displayOnMATLAB(stream);
        }

        return this->createAllMatrices;
    }
};
