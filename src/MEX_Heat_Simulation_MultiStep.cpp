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

    /*Inputs required : T0, tissueSize, tissueProperties, BC, Flux, ambientTemp, sensorLocations, beamWaist, time, laserPose, laserPower, 
       OPTIONAL       : (useAllCPUs), (silentMode), (layers), (Nn1d), (alpha)*/
    enum VarPlacement {TEMPERATURE, TISSUE_SIZE, TISSUE_PROP, BC, FLUX, AMB_TEMP, SENSOR_LOC, BEAM_WAIST, TIME, LASER_POSE, LASER_POWER,
                        USE_ALL_CPUS, SILENT_MODE, LAYERS, NN1D, ALPHA};

    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    FEM_Simulator* simulator;
    std::ostringstream stream;
    bool silentMode = false;
    bool useAllCPUs = true;
    bool createAllMatrices = true;
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
    std::vector<std::vector<std::vector<float>>> convertMatlabArrayTo3DVector(const matlab::data::Array& matlabArray) {
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

    /** @brief Converts a matlab array to a std::vector
    *
    *
    */
    std::vector<float> convertMatlabArrayTo1DVector(const matlab::data::Array& matlabArray) {       
        size_t numCols = matlabArray.getDimensions()[1];
        std::vector<float> result;
        result.reserve(numCols);
        for (size_t i = 0; i < numCols; ++i) {
            result.push_back(static_cast<float>(matlabArray[0][i]));
        }
        return result;
    }


    /** @brief Convert a matlab array to a Eigen::MatrixXf
    *
    *
    */
    Eigen::MatrixXf convertMatlabArrayToEigenMatrix(const matlab::data::Array& matlabArray) {
        const size_t rows = matlabArray.getDimensions()[0];
        const size_t cols = matlabArray.getDimensions()[1];

        Eigen::MatrixXf mat(rows, cols);
        for (size_t j = 0; j < cols; ++j) {
            for (size_t i = 0; i < rows; ++i) {
                mat(i, j) = static_cast<float>(matlabArray[i][j]);
            }
        }
        return mat;
    }

    /** @brief Convert a 3D std::vector to a matlab array
    *
    * 
    */
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


    /** @brief This is the gateway routine for the MEX-file. 
    *
    * This function actually takes the inputs from matlab and performs the heat simulation steps
    * 
    * @param outputs - list of outputs matlab expects
    * @param inputs - list of inputs matlab provideds
    */
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        auto startTime = std::chrono::high_resolution_clock::now();
        auto stopTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
        checkArguments(outputs, inputs);
        stream.str("");
        stream << "MEX: Start of MEX function" << std::endl;
        displayOnMATLAB(stream);
        std::vector<float> timeVec;
        std::vector<float> laserPower;
        Eigen::MatrixXf laserPose;
        float beamWaist;
        try {

            // Have to convert T0 and FluenceRate to std::vector<<<float>>>
            std::vector<std::vector<std::vector<float>>> T0 = convertMatlabArrayTo3DVector(inputs[VarPlacement::TEMPERATURE]);
            stopTime = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
            stream << "MEX: Converted MATLAB Matrices:  " << duration.count() / 1000000.0 << " seconds" << std::endl;
            displayOnMATLAB(stream);

            // Get tissue size
            float tissueSize[3];
            tissueSize[0] = inputs[VarPlacement::TISSUE_SIZE][0];
            tissueSize[1] = inputs[VarPlacement::TISSUE_SIZE][1];
            tissueSize[2] = inputs[VarPlacement::TISSUE_SIZE][2];

            // Extract tissue properties
            float MUA = inputs[VarPlacement::TISSUE_PROP][0];
            float TC = inputs[VarPlacement::TISSUE_PROP][1];
            float VHC = inputs[VarPlacement::TISSUE_PROP][2];
            float HTC = inputs[VarPlacement::TISSUE_PROP][3];

            // Get boundary conditions
            int boundaryType[6] = { 0,0,0,0,0,0 };
            //stream << "Boundary Conditions: ";
            for (int i = 0; i < 6; i++) {
                boundaryType[i] = inputs[VarPlacement::BC][i];
            }

            // get heatFlux condition
            float heatFlux = inputs[VarPlacement::FLUX][0];
            // get ambient temperature
            float ambientTemp = inputs[VarPlacement::AMB_TEMP][0];

            // beam waist
            beamWaist = inputs[VarPlacement::BEAM_WAIST][0];
            stream << "MEX: Setting time dependent variables" << std::endl;
            displayOnMATLAB(stream);
            // get time varying inputs
            timeVec = convertMatlabArrayTo1DVector(inputs[VarPlacement::TIME]);
            laserPower = convertMatlabArrayTo1DVector(inputs[VarPlacement::LASER_POWER]);
            laserPose = convertMatlabArrayToEigenMatrix(inputs[VarPlacement::LASER_POSE]);

            stream << "MEX: Apply inputs to simulator" << std::endl;
            displayOnMATLAB(stream);

            /* SET ALL PARAMETERS NECESSARY BEFORE CONSTRUCTING ELEMENTS*/
            // Set the type of basis functions we are using by setting nodes per dimension of an 
            this->simulator->Nn1d = Nn1d;
            this->simulator->setTemp(T0);
            this->simulator->setTissueSize(tissueSize);
            // set the layer info
            this->simulator->setLayer(layerHeight, elemsInLayer);
            // set the tissue properties
            this->simulator->setMUA(MUA);
            this->simulator->setTC(TC);
            this->simulator->setVHC(VHC);
            this->simulator->setHTC(HTC);
            // set boundary conditions
            this->simulator->setBoundaryConditions(boundaryType);
            // set heatFlux
            this->simulator->setFlux(heatFlux);
            // set ambient temperature
            this->simulator->setAmbientTemp(ambientTemp);
        }
        catch (const std::exception& e) {
            stream << "MEX: Error in Setup: " << std::endl;
            displayOnMATLAB(stream);
            displayError(e.what());
            return;
        }

        // Set sensor locations
        auto sensorTempsInput = inputs[VarPlacement::SENSOR_LOC];
        std::vector<std::array<float, 3>> sensorTemps;
        for (int s = 0; s < sensorTempsInput.getDimensions()[0]; s++) {
            sensorTemps.push_back({ sensorTempsInput[s][0],sensorTempsInput[s][1] ,sensorTempsInput[s][2] });
        }
        try {
            this->simulator->setSensorLocations(sensorTemps);
        }
        catch (const std::exception& e) {
            stream << "MEX: Error setting sensor locations " << std::endl;
            displayOnMATLAB(stream);
            displayError(e.what());
            return;
        }

        stopTime = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
        stream << "MEX: Set all simulation parameters:  " << duration.count() / 1000000.0 << " seconds" << std::endl;
        displayOnMATLAB(stream);

        // Set parallelization
        Eigen::setNbThreads(1);
#ifdef _OPENMP
        stream << "MEX: OPEMMP Enabled" << std::endl;
        displayOnMATLAB(stream);
        if (useAllCPUs) { //useAllCPUs is true
            Eigen::setNbThreads(omp_get_num_procs() / 2);
        }
#endif
        stream << "MEX: Number of threads: " << Eigen::nbThreads() << std::endl;
        displayOnMATLAB(stream);


        /* Initialize the model for time = 0 or the first index and build the matrices*/
        try {
            this->simulator->deltaT = timeVec[1] - timeVec[0]; // set deltaT
            this->simulator->setFluenceRate(laserPose.col(0), laserPower[0], beamWaist); // set fluence rate
            this->simulator->initializeModel(); // initialize model
            /*stream << "Global matrices created" << std::endl;
            displayOnMATLAB(stream);*/
        }
        catch (const std::exception& e) {
            stream << "MEX: Error in createKMF() " << std::endl;
            displayOnMATLAB(stream);
            displayError(e.what());
            return;
        }
        stopTime = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
        stream << "MEX: Matrices Built:  " << duration.count() / 1000000.0 << " seconds" << std::endl;
        displayOnMATLAB(stream);

        /* Now perform time stepping with singele steps */
        int numSteps = timeVec.size() - 1; // if time is of size 2, then we only have 1 step.
        this->simulator->initializeSensorTemps(numSteps);
        this->simulator->updateTemperatureSensors(0);
        for (int t = 1; t <= numSteps; t++) {
            // we are simulating going from step t-1 to t. In the first case this is going from t[0] to t[1]. 
            try {
                this->simulator->deltaT = timeVec[t] - timeVec[t - 1]; // set deltaT
                // fluence rate is set based on parameters at time = t. This is because for single step
                // the explicit portion was actually calculated during the previous call, or during initializeModel()
                // So now we are really calculating the implicit step (backwards euler or crank-nicolson) which requires future input
                this->simulator->setFluenceRate(laserPose.col(t), laserPower[t], beamWaist);
                this->simulator->singleStepCPU();
                this->simulator->updateTemperatureSensors(t);
            }
            catch (const std::exception& e) {
                stream << "MEX: Error in performTimeStepping() " << std::endl;
                displayOnMATLAB(stream);
                displayError(e.what());
                return;
            }
            stream << "Step " << t << " of " << numSteps << std::endl;
            displayOnMATLAB(stream);
        }
        stopTime = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
        stream << "MEX: Time Stepping Complete:  " << duration.count() / 1000000.0 << " seconds" << std::endl;
        displayOnMATLAB(stream);

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

        stopTime = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
        stream << "MEX: End of MEX Function:  " << duration.count() / 1000000.0 << std::endl;
        displayOnMATLAB(stream);
    }

    /** @brief This function makes sure that user has provided the proper inputs
    * 
    * This function will check the number, dimensions, and type of each input to make sure they were passed in properly.
    * Additionally, it will make sure the user is receiving the right number of outputs.
    * 
    * @param outputs - list of outputs matlab expects
    * @param inputs - list of inputs matlab provideds
    * 
    */
    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if (inputs.size() < 11) {
            displayError("At least 11 inputs required: T0, tissueSize,"
                "tissueProperties, BC, Flux, ambientTemp, sensorLocations, beamWaist, time, laserPose, laserPower, (useAllCPUs), (silentMode), (layers), (Nn1d), (createAllMatrices) ");
        }
        if (outputs.size() > 2) {
            displayError("Too many outputs specified.");
        }
        if (outputs.size() < 2) {
            displayError("Not enough outputs specified.");
        }
        // Check Temp Input
        if (inputs[VarPlacement::TEMPERATURE].getType() != matlab::data::ArrayType::SINGLE) {
            displayError("T0 must be an Array of type Single.");
        }
        if ((inputs[VarPlacement::TISSUE_SIZE].getDimensions()[0] != 3) || (inputs[VarPlacement::TISSUE_SIZE].getDimensions()[1] != 1)) {
            displayError("Tissue Size must be 3 x 1");
        }
        if ((inputs[VarPlacement::TISSUE_PROP].getDimensions()[0] != 4) || (inputs[VarPlacement::TISSUE_PROP].getDimensions()[1] != 1)) {
            displayError("Tissue Properties must be 4 x 1: MUA, TC, VHC, HTC");
        }
        if ((inputs[VarPlacement::BC].getDimensions()[0] != 6) || (inputs[VarPlacement::BC].getDimensions()[1] != 1)) {
            displayError("Boundary Conditions Size must be 6 x 1");
        }
        if ((inputs[VarPlacement::BC].getType() != matlab::data::ArrayType::INT32)) {
            displayError("Boundary Conditions must be an int");
        }
        if ((inputs[VarPlacement::SENSOR_LOC].getDimensions()[1] != 3)) {
            displayError("Sensor Locations must be n x 3");
        }
        if ((inputs[VarPlacement::TIME].getDimensions()[0] != 1)) {
            stream << "Time size in first dimension: " << inputs[VarPlacement::TIME].getDimensions()[0] << std::endl;
            displayOnMATLAB(stream);
            displayError("Time should be a row vector 1xn");
        }
        if ((inputs[VarPlacement::TIME].getDimensions()[1] < 2)) {
            displayError("Time should have at least 2 elements");
        }
        if ((inputs[VarPlacement::LASER_POWER].getDimensions()[0] != 1)) {
            displayError("LaserPower should be a row vector 1xn");
        }
        if ((inputs[VarPlacement::LASER_POWER].getDimensions()[1] < 2)) {
            displayError("LaserPower should have at least 2 elements");
        }
        if ((inputs[VarPlacement::LASER_POSE].getDimensions()[0] != 6)) {
            displayError("LaserPose should be a 6xn vector");
        }
        if ((inputs[VarPlacement::LASER_POSE].getDimensions()[1] < 2)) {
            displayError("LaserPose should have at least 2 columns");
        }
        if ((inputs[VarPlacement::TIME].getDimensions()[1] != inputs[VarPlacement::LASER_POWER].getDimensions()[1]) ||
            (inputs[VarPlacement::TIME].getDimensions()[1] != inputs[VarPlacement::LASER_POSE].getDimensions()[1])) {
            displayError("Time vector, laserPower vector, and laserPose matrix should have the same number of columns");
        }
        if (inputs.size() > VarPlacement::USE_ALL_CPUS) {
            useAllCPUs = inputs[VarPlacement::USE_ALL_CPUS][0];
        }
        if (inputs.size() > VarPlacement::SILENT_MODE) {
            this->silentMode = inputs[VarPlacement::SILENT_MODE][0];
            this->simulator->silentMode = this->silentMode;
        }
        if (inputs.size() > VarPlacement::LAYERS) {
            layerHeight = inputs[VarPlacement::LAYERS][0];
            elemsInLayer = int(inputs[VarPlacement::LAYERS][1]);
        }
        else {
            elemsInLayer = inputs[VarPlacement::TEMPERATURE].getDimensions()[2];
            layerHeight = inputs[VarPlacement::TISSUE_SIZE][2];
        }
        if (inputs.size() > VarPlacement::NN1D) {
            Nn1d = inputs[VarPlacement::NN1D][0];
        }
        if (inputs.size() > VarPlacement::ALPHA) {
            this->simulator->alpha = inputs[VarPlacement::ALPHA][0];
        }
    }
};
