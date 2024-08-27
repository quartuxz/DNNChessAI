#include "NeuralNetwork.cuh"
#include <random>
#include <math.h>
#include <stdexcept>
#include <chrono>
#include <iostream>



#include "device_launch_parameters.h"

#include <stdio.h>


std::default_random_engine NNInitialization::engine = std::default_random_engine(static_cast<long unsigned int>(time(0)));


#define THREADS_PER_BLOCK 512

#define CUDA_CHECK(call) {cudaError_t cudaStatus = call; if(cudaStatus != cudaSuccess){ fprintf(stderr, "CUDA ERROR: %s, on file: %s, in function: %s, in line: %d \n",cudaGetErrorString(cudaStatus),__FILE__,__func__, __LINE__);}}



void NeuralNetwork::m_initializeGpuMem()
{

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//initialize the GPU memory
	for (size_t i = 0; i < m_hiddenLayers.size() + 1; i++) {
		Layer* layer;

		if (i < m_hiddenLayers.size()) {
			layer = &m_hiddenLayers[i];
		}
		else {
			layer = &m_outputLayer;
		}
		size_t allSynapsesSize = layer->prevSize * layer->size;
		GPUDataLayer data = std::make_tuple<float*, float*, unsigned int*>(0, 0, 0);

		cudaStatus = cudaMalloc((void**)&std::get<0>(data), allSynapsesSize * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&std::get<1>(data), layer->size * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMalloc((void**)&std::get<2>(data), 3 * sizeof(unsigned int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}
		m_GPULayers.push_back(data);

		std::pair<float*, float*> data2 = std::pair<float*, float*>(0, 0);

		cudaStatus = cudaMalloc((void**)&std::get<0>(data2), allSynapsesSize * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}
		

		cudaStatus = cudaMemset(std::get<0>(data2),0, allSynapsesSize * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemset failed!");
		}
		cudaStatus = cudaMalloc((void**)&std::get<1>(data2), layer->size * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		cudaStatus = cudaMemset(std::get<1>(data2), 0, layer->size * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemset failed!");
		}

		m_GPUMomentum.push_back(data2);

	}
}

//we build up the neural network
NeuralNetwork::NeuralNetwork(Topology top, NNInitialization init, LearningSchedule ls, NNActivation act):
	m_top(top),
	m_act(act),
	m_learningSched(ls),
	m_outputLayer(init,top.back(),top[top.size()-2],true)
{
	//we create the layers
	for (size_t i = 0; i < top.size()-2; i++)
	{
		m_hiddenLayers.push_back(Layer(init, top[i + 1], top[i]));
	}

	m_initializeGpuMem();
	//TODO: optimize this to not run during deserializing 
	m_updateGPUMem();
}


size_t getEpochFStr(const std::string &str) {
	std::string currentNumber;
	
	size_t retval;

	for (char current : str) {
		if (current == '\n') {
			break;
		}
		if (current == ' ') {
			retval = std::atoi(currentNumber.c_str());
			currentNumber = "";
			continue;
		}
		else {
			currentNumber.push_back(current);
		}

	}
	return retval;
}

Topology getTopology(const std::string& str) {
	std::string currentNumber;

	Topology retval;

	size_t line = 0;

	for (char current : str) {
		if (current == '\n') {
			if (line == 1) {
				break;
			}
			line++;
		}
		if (line != 1) {
			continue;
		}
		if (current == ' ') {
			retval.push_back(std::atoi(currentNumber.c_str()));
			currentNumber = "";
			continue;
		}
		else {
			currentNumber.push_back(current);
		}
		
	}
	return retval;
}

NeuralNetwork::NeuralNetwork(const std::string& str) :
	NeuralNetwork(getTopology(str),NNInitialization(),LearningSchedule())
{
	m_epoch = getEpochFStr(str);
	std::string currentNumber;

	std::vector<Layer*> allLayers;

	for (auto& layer : m_hiddenLayers) {
		allLayers.push_back(&layer);
	}
	allLayers.push_back(&m_outputLayer);
	
	//line of the string being read
	unsigned int line = 0;

	//orienting indexes for creating the NN with the corresponding values as read.
	unsigned int layerN = 0, neuronN = 0, synapseN = 0;
	bool isBias = true;
	for (char current : str) {
		//the first line was already read for topology information.
		if (current == '\n') {
			if (line >= 2) {
				break;
			}
			line++;
			continue;
		}
		if (line < 2) {
			continue;
		}
		

		if (current == ' ') {
			float parsedNumber = std::atof(currentNumber.c_str());
			if (isBias) {
				allLayers[layerN]->biases[neuronN] = parsedNumber;
				isBias = false;
			}
			else {
				allLayers[layerN]->weights[neuronN][synapseN] = parsedNumber;
				currentNumber = "";
				synapseN++;
			}
			continue;
		}
		else if (current == '/') {
			isBias = true;
			neuronN++;
			synapseN = 0;
		}
		else if (current == ',') {
			layerN++;
			neuronN = 0;
		}
		else {
			currentNumber.push_back(current);
		}

	}
	m_updateGPUMem();
}

size_t NeuralNetwork::accumulateInstanceForBackprop(const std::vector<std::vector<float>>& instaceOutputDif)
{

	m_accumulatedInstancesForBP.insert(m_accumulatedInstancesForBP.end(), instaceOutputDif.begin(), instaceOutputDif.end());

	return m_accumulatedInstancesForBP.size();
}



void NeuralNetwork::addRandomWeights()
{

	for (size_t i = 0; i < m_hiddenLayers.size() + 1; i++)
	{
		Layer* layer;
		if (i < m_hiddenLayers.size()) {
			layer = &m_hiddenLayers[i];
		}
		else {
			layer = &m_outputLayer;
		}

		layer->randomizeParameters(m_learningSched);
	}
	m_updateGPUMem();
}


NeuralNetwork::NeuralNetwork(NeuralNetwork &other):
	m_act(other.m_act),
	m_learningSched(other.m_learningSched),
	m_top(other.m_top),
	m_batchN(other.m_batchN)
{
	std::cout << "USED NN COPY CONSTRUCTOR!" << std::endl;
	other.m_updateRAM();
	m_hiddenLayers = other.m_hiddenLayers;
	m_outputLayer = other.m_outputLayer;
	m_initializeGpuMem();
	m_updateGPUMem();
}

NeuralNetwork& NeuralNetwork::operator=(NeuralNetwork other)
{
	swap(*this,other);
	return *this;
}



__global__ void calculateZErrorKernel(float **thisZError, float *thisBiasAcc,  float **thisZ, float **thisA, float **nextZError,  const float *weightMatrix, const unsigned int* extraParamsThisLayer, const unsigned int *extraParamsNextLayer)
{
	int instance = blockIdx.y;
	int neuron = threadIdx.x + blockDim.x * blockIdx.x;
	if (neuron < extraParamsThisLayer[2]) {


		const unsigned int matrixWidth = extraParamsThisLayer[2];
		const unsigned int matrixHeight = extraParamsNextLayer[2];
		float errorWeightSum = 0;
		for (size_t i = 0; i < matrixHeight; i++) {
			errorWeightSum += weightMatrix[neuron + matrixWidth * i]*nextZError[instance][i];
		}
		float deriv = 0;
		//if (extraParamsThisLayer[0] == 2) {

		//	deriv = thisA[instance][neuron] * (1 - thisA[instance][neuron]);
		//}
		//else if (extraParamsThisLayer[0] == 3) {
			float sigmoid = 1 / (1 + exp(-thisZ[instance][neuron]));
			deriv = thisA[instance][neuron]+ sigmoid *(1-thisA[instance][neuron]);
		//}


		thisZError[instance][neuron] = errorWeightSum * deriv;



		atomicAdd(thisBiasAcc+neuron, thisZError[instance][neuron]);
	}
}

__global__ void calculateWeightGradientsKernel(float *weightMatrixAcc, float **thisZError, float **prevA, const unsigned int *extraParams) {
	int instance = blockIdx.y;
	int neuron = threadIdx.x + blockDim.x * blockIdx.x;
	if (neuron < extraParams[2]) {
		for (size_t i = 0; i < extraParams[1]; i++) {
			atomicAdd(weightMatrixAcc+ i + neuron * extraParams[1], prevA[instance][i] * thisZError[instance][neuron]);
		}
	}
}

//TODO: implement adam optimizer.
//currently uses Momentum+SGD
__global__ void addAverageWeightsAndBiasesKernel(float *weights, const float *weightsAcc, float *biases, const float *biasesAcc,const float* instancesAndLearningRate,const unsigned int* extraParams, float*momentumWeights,float* momentumBiases) {
	int neuron = threadIdx.x + blockDim.x * blockIdx.x;
	if (neuron < extraParams[2]) {
		for (size_t i = 0; i < extraParams[1]; i++) {
			momentumWeights[i + neuron * extraParams[1]] = instancesAndLearningRate[2] * momentumWeights[i + neuron * extraParams[1]] - instancesAndLearningRate[1] * (weightsAcc[i + neuron * extraParams[1]] / instancesAndLearningRate[0]);
			atomicAdd(weights + i + neuron * extraParams[1], momentumWeights[i + neuron * extraParams[1]]);
		}
		momentumBiases[neuron] = instancesAndLearningRate[2] * momentumBiases[neuron] - instancesAndLearningRate[1] * (biasesAcc[neuron] / instancesAndLearningRate[0]);
		atomicAdd(biases + neuron, momentumBiases[neuron]);
	}
}


void NeuralNetwork::increaseEpoch()
{
	m_epoch++;
}

void NeuralNetwork::setEpoch(size_t epoch)
{
	m_epoch = epoch;
}

size_t NeuralNetwork::getEpoch() const
{
	return m_epoch;
}

void NeuralNetwork::backpropagateGPU(std::vector<std::vector<float>>& dCost_dOutput_forInstances, size_t numberOfStreams)
{

	m_instancesInBatch = 0;


	if (dCost_dOutput_forInstances.empty()) {
		return;
	}

	std::vector<float*> weightAcc;
	std::vector<float*> biasAcc;

	cudaError_t cudaStatus = cudaSetDevice(0);
	for (size_t i = 0; i < m_hiddenLayers.size() + 1; i++) {
		const Layer* layer = (i < m_hiddenLayers.size() ? &m_hiddenLayers[i] : &m_outputLayer);

		size_t allWeightsSize = layer->prevSize * layer->size;

		float* weights = new float[allWeightsSize];
		float* biases = new float[layer->size];

		if (i < m_hiddenLayers.size()) {
			layer = &m_hiddenLayers[i];
		}
		else {
			layer = &m_outputLayer;
		}
		for (size_t neuron = 0; neuron < layer->size; neuron++) {
			for (size_t prevNeuron = 0; prevNeuron < layer->prevSize; prevNeuron++) {
				weights[prevNeuron + neuron * layer->prevSize] = 0;
			}
			biases[neuron] = 0;
		}

		weightAcc.push_back(0);
		biasAcc.push_back(0);

		cudaStatus = cudaMalloc((void**)&weightAcc.back(), allWeightsSize * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed! weightAcc, backpropagateGPU");
			system("pause");
		}

		cudaStatus = cudaMalloc((void**)&biasAcc.back(), layer->size * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!, biasAcc, backpropagateGPU");
			system("pause");
		}



		cudaStatus = cudaMemcpy(weightAcc.back(), weights, allWeightsSize * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!, weightAcc, backpropagateGPU");
			system("pause");
		}

		cudaStatus = cudaMemcpy(biasAcc.back(), biases, layer->size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!, biasAcc, backpropagateGPU");
			system("pause");
		}

		delete[] weights;
		delete[] biases;
	}


	
	//first axis is layer second axis is instance, third axis is neuron.
	std::vector<float**> dCost_dErrorL;

	std::vector<float*> allNeuronsD;

	//reverse order, last layer first
	for (int o = m_hiddenLayers.size() - 1; o >= 0; o--) {




		if (o == m_hiddenLayers.size() - 1) {
			dCost_dErrorL.push_back(0);

			float** RAMDCost_dError = new float* [dCost_dOutput_forInstances.size()];
			CUDA_CHECK(cudaMalloc((void**)&dCost_dErrorL.back(), dCost_dOutput_forInstances.size()*sizeof(float*)));
			for (size_t i = 0; i < dCost_dOutput_forInstances.size(); i++) {
				float* outputsGPU = 0;
				CUDA_CHECK(cudaMalloc((void**)&outputsGPU, dCost_dOutput_forInstances[i].size() * sizeof(float)));
				CUDA_CHECK(cudaMemcpy(outputsGPU, dCost_dOutput_forInstances[i].data(), dCost_dOutput_forInstances[i].size() * sizeof(float), cudaMemcpyHostToDevice));
				allNeuronsD.push_back(outputsGPU);
				RAMDCost_dError[i] = outputsGPU;
			}

			CUDA_CHECK(cudaMemcpy(dCost_dErrorL.back(), RAMDCost_dError, dCost_dOutput_forInstances.size() * sizeof(float*),cudaMemcpyHostToDevice));
			delete[]RAMDCost_dError;
		}

		//error in the next layer for the current instance
		float** nextZError = dCost_dErrorL.back();
		dCost_dErrorL.push_back(0);
		{
			CUDA_CHECK(cudaMalloc((void**)&dCost_dErrorL.back(), dCost_dOutput_forInstances.size() * sizeof(float*)));
			float** nextLayerZeroes = new float* [dCost_dOutput_forInstances.size()];
			for (size_t i = 0; i < dCost_dOutput_forInstances.size(); i++) {
				float* layerZeroesGPU = 0;
				CUDA_CHECK(cudaMalloc((void**)&layerZeroesGPU, m_hiddenLayers[o].size * sizeof(float)));
				//CUDA_CHECK(cudaMemset(layerZeroesGPU,0, m_hiddenLayers[o].size*sizeof(float)));
				nextLayerZeroes[i] = layerZeroesGPU;
				allNeuronsD.push_back(layerZeroesGPU);
			}

			CUDA_CHECK(cudaMemcpy(dCost_dErrorL.back(), nextLayerZeroes, dCost_dOutput_forInstances.size() * sizeof(float*), cudaMemcpyHostToDevice));
			delete[]nextLayerZeroes;
		}




		float** savedZValuesInstance  = new float*[m_savedZValues.size()];
		float** savedZValuesInstanceGPU = 0;
		CUDA_CHECK(cudaMalloc((void**)&savedZValuesInstanceGPU, m_savedZValues.size()*sizeof(float*)));
		for (size_t p = 0; p < m_savedZValues.size(); p++) {
			savedZValuesInstance[p] = m_savedZValues[p][o + 1];
		}
		CUDA_CHECK(cudaMemcpy(savedZValuesInstanceGPU,savedZValuesInstance, m_savedZValues.size() * sizeof(float*),cudaMemcpyHostToDevice));
		delete[] savedZValuesInstance;

		float** savedAValuesInstance = new float* [m_savedAValues.size()];
		float** savedAValuesInstanceGPU = 0;
		CUDA_CHECK(cudaMalloc((void**)&savedAValuesInstanceGPU, m_savedAValues.size() * sizeof(float*)));
		for (size_t p = 0; p < m_savedAValues.size(); p++) {
			savedAValuesInstance[p] = m_savedAValues[p][o + 1];
		}
		CUDA_CHECK(cudaMemcpy(savedAValuesInstanceGPU, savedAValuesInstance, m_savedAValues.size() * sizeof(float*), cudaMemcpyHostToDevice));
		delete [] savedAValuesInstance;



		unsigned int blockSizeX = std::max((unsigned int)std::ceilf(m_hiddenLayers[o].size / THREADS_PER_BLOCK), (unsigned int)1);
		unsigned int threadSizeX = std::min((size_t)THREADS_PER_BLOCK, m_hiddenLayers[o].size);


		dim3 dimGrid(blockSizeX, dCost_dOutput_forInstances.size(),1);

		// Launch a kernel on the GPU with one thread for each element.
		//accumulate biases for this instance
		//std::cout << "Layer: " << o << "." << std::endl;
		calculateZErrorKernel <<< dimGrid, threadSizeX >>> (dCost_dErrorL.back(), biasAcc[o], savedZValuesInstanceGPU, savedAValuesInstanceGPU, nextZError, std::get<0>(m_GPULayers[o + 1]), std::get<2>(m_GPULayers[o]), std::get<2>(m_GPULayers[o + 1]));

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "calculateZErrorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			system("pause");
		}

		CUDA_CHECK(cudaDeviceSynchronize());

		float** savedAValuesInstance2 = new float* [m_savedAValues.size()];
		float** savedAValuesInstanceGPU2 = 0;
		CUDA_CHECK(cudaMalloc((void**)&savedAValuesInstanceGPU2, m_savedAValues.size() * sizeof(float*)));
		for (size_t p = 0; p < m_savedAValues.size(); p++) {
			savedAValuesInstance2[p] = m_savedAValues[p][o];
		}
		CUDA_CHECK(cudaMemcpy(savedAValuesInstanceGPU2, savedAValuesInstance2, m_savedAValues.size() * sizeof(float*), cudaMemcpyHostToDevice));
		delete[] savedAValuesInstance2;


		//accumulate weights for this instance
		calculateWeightGradientsKernel <<< dimGrid, threadSizeX >>> (weightAcc[o], dCost_dErrorL.back(), savedAValuesInstanceGPU2, std::get<2>(m_GPULayers[o]));

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "calculateWeightGradientsKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			system("pause");
		}
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		CUDA_CHECK(cudaDeviceSynchronize());

		cudaFree(savedAValuesInstanceGPU);
		cudaFree(savedZValuesInstanceGPU);
		cudaFree(savedAValuesInstanceGPU2);
	}

	for (auto instanceLayer : allNeuronsD) {
		cudaFree(instanceLayer);
	}

	allNeuronsD.clear();

	for (auto instance : dCost_dErrorL) {
		cudaFree(instance);
	}
	dCost_dErrorL.clear();

	float* instancesAndLearningRate = new float[3];
	instancesAndLearningRate[0] = dCost_dOutput_forInstances.size();
	instancesAndLearningRate[1] = m_learningSched.getLearningRate(m_epoch);
	instancesAndLearningRate[2] = m_learningSched.momentum;
	float *instancesAndLearningRateGPU=0;
	cudaStatus = cudaMalloc((void**)&instancesAndLearningRateGPU, 3* sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(instancesAndLearningRateGPU, instancesAndLearningRate, 3* sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}


	for (size_t i = 0; i < m_hiddenLayers.size() + 1; i++) {
		const Layer* layer = (i < m_hiddenLayers.size() ? &m_hiddenLayers[i] : &m_outputLayer);
		unsigned int blockSize = std::max((unsigned int)std::ceilf(layer->size / THREADS_PER_BLOCK), (unsigned int)1);
		unsigned int threadSize = std::min((size_t)THREADS_PER_BLOCK, layer->size);
		addAverageWeightsAndBiasesKernel <<< blockSize, threadSize >>> (std::get<0>(m_GPULayers[i]), weightAcc[i], std::get<1>(m_GPULayers[i]), biasAcc[i], instancesAndLearningRateGPU, std::get<2>(m_GPULayers[i]), std::get<0>(m_GPUMomentum[i]), std::get<1>(m_GPUMomentum[i]));
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addAverageAndBiasesKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			system("pause");
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addAverageAndBiasesKernel!\n", cudaStatus);
			system("pause");
		}
		
	}

	delete instancesAndLearningRate;
	cudaFree(instancesAndLearningRateGPU);


	for (auto weight : weightAcc) {
		cudaFree(weight);
	}
	for (auto bias : biasAcc) {
		cudaFree(bias);
	}

	clearTrainingData();
	m_batchN++;
}

void NeuralNetwork::backpropagateGPU( size_t numberOfStreams)
{
	backpropagateGPU(m_accumulatedInstancesForBP,numberOfStreams);

}

void NeuralNetwork::startRecording()
{

	m_recording = true;
}

void NeuralNetwork::endRecording()
{
	m_instanceN = 0;
	m_recording = false;

}

void NeuralNetwork::selectAndDiscardRest(unsigned int selected, bool selectAll)
{
	m_instancesInBatch++;

	if (selectAll) {
		m_savedAValues.insert(m_savedAValues.end(),m_intermediateAValues.begin(), m_intermediateAValues.end());
		m_savedZValues.insert(m_savedZValues.end(), m_intermediateZValues.begin(), m_intermediateZValues.end());
	} else {
		m_savedAValues.push_back(m_intermediateAValues[selected]);
		m_savedZValues.push_back(m_intermediateZValues[selected]);


		for (size_t i = 0; i < m_intermediateAValues.size(); i++) {
			if (i != selected) {
				for (auto val : m_intermediateAValues[i]) {
					CUDA_CHECK(cudaFree(val));
				}
			}
		}
		for (size_t i = 0; i < m_intermediateZValues.size(); i++) {
			if (i != selected) {
				for (auto val : m_intermediateZValues[i]) {
					CUDA_CHECK(cudaFree(val));
				}
			}
		}
	}



	m_intermediateAValues.clear();
	m_intermediateZValues.clear();

}

void NeuralNetwork::clearTrainingData()
{
	m_accumulatedInstancesForBP.clear();
	for (auto& layer : m_savedAValues) {
		for (auto val : layer) {
			cudaFree(val);
		}
	}
	m_savedAValues.clear();
	for (auto& layer : m_savedZValues) {
		for (auto val : layer) {
			cudaFree(val);
		}
	}
	m_savedZValues.clear();
}


#include <sstream>

std::vector<float> NeuralNetwork::forwardPassCPU(const std::vector<float>& input)
{
	m_updateRAM();
	if (m_top[0] != input.size()) {
		std::stringstream ss;
		ss << "input layer is not the same size as the parameters passed: " << m_top[0] << " vs " << input.size() << std::endl;
		throw std::invalid_argument(ss.str());
	}

	std::vector<float> prevResult = input;
	std::vector<float> thisResult;

	for (size_t i = 0; i < m_hiddenLayers.size() + 1; i++)
	{
		const Layer* layer;
		if (i < m_hiddenLayers.size()) {
			layer = &m_hiddenLayers[i];
		}
		else {
			layer = &m_outputLayer;
		}

		for (size_t o = 0; o < layer->size; o++) {
			auto& synapses = layer->weights[o];
			float weightedSum = 0;
			for (size_t p = 0; p < layer->prevSize;p++) {
				weightedSum += synapses[p] * prevResult[p];
			}
			thisResult.push_back(layer->act.func(weightedSum + layer->biases[o]));
		}
		prevResult = thisResult;
	}


	return thisResult;
}








void NeuralNetwork::m_updateGPUMem()
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}


	for (size_t i = 0; i < m_hiddenLayers.size() + 1; i++) {
		const Layer* layer = (i < m_hiddenLayers.size() ? &m_hiddenLayers[i] : &m_outputLayer);

		size_t allWeightsSize = layer->prevSize * layer->size;

		float* weights = new float[allWeightsSize];
		float* biases = new float[layer->size];
		unsigned int* extraParams = new unsigned int[3];
		extraParams[0] = (unsigned int)m_act.actType;
		extraParams[1] = (unsigned int)layer->prevSize;
		extraParams[2] = (unsigned int)layer->size;

		if (i < m_hiddenLayers.size()) {
			layer = &m_hiddenLayers[i];
		}
		else {
			layer = &m_outputLayer;
		}
		for (size_t neuron = 0; neuron < layer->size; neuron++) {
			for (size_t prevNeuron = 0; prevNeuron < layer->prevSize; prevNeuron++) {
				weights[prevNeuron + neuron * layer->prevSize] = layer->weights[neuron][prevNeuron];
			}
			biases[neuron] = layer->biases[neuron];
		}
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(std::get<0>(m_GPULayers[i]), weights, allWeightsSize * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(std::get<1>(m_GPULayers[i]), biases, layer->size * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(std::get<2>(m_GPULayers[i]), extraParams, 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed! extraParams, m_updateGPUmem()");
		}
		delete[] weights;
		delete[] biases;
		delete[] extraParams;
	}

}

void NeuralNetwork::m_updateRAM()
{
	cudaError_t cudaStatus = cudaSetDevice(0);
	for (size_t i = 0; i < m_hiddenLayers.size() + 1; i++) {
		Layer* layer = (i < m_hiddenLayers.size() ? &m_hiddenLayers[i] : &m_outputLayer);

		size_t allWeightsSize = layer->prevSize * layer->size;

		float* weights = new float[allWeightsSize];
		float* biases = new float[layer->size];


		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(weights, std::get<0>(m_GPULayers[i]), allWeightsSize * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(biases, std::get<1>(m_GPULayers[i]), layer->size * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		if (i < m_hiddenLayers.size()) {
			layer = &m_hiddenLayers[i];
		}
		else {
			layer = &m_outputLayer;
		}
		for (size_t neuron = 0; neuron < layer->size; neuron++) {
			for (size_t prevNeuron = 0; prevNeuron < layer->prevSize; prevNeuron++) {
				layer->weights[neuron][prevNeuron] = weights[prevNeuron + neuron * layer->prevSize];
			}
			layer->biases[neuron] = biases[neuron];
		}
		delete[] weights;
		delete[] biases;
	}
}




__global__ void forwardPassLayerKernel(float* zValues, float* VecOut, float* VecIn, const float* weights, const float* biases, const unsigned int* extraParams, unsigned int layerSize, unsigned int inputLayerSize)
{
	//takes into account what block partition it is in(if more than maximum amount of neurons is being processed in layer) with block.x
	int neuronAbs = threadIdx.x + blockDim.x * blockIdx.x;
	//takes into account what instance it is processing with block.y
	int neuron = neuronAbs + blockIdx.y * layerSize;
	if (neuronAbs < extraParams[2]) {

		float sum = 0;
		for (size_t i = 0; i < extraParams[1]; i++) {
			sum += weights[i + neuronAbs * extraParams[1]] * VecIn[i + blockIdx.y * inputLayerSize];
		}
		zValues[neuron] = sum + biases[neuronAbs];
		if (extraParams[0] == 1) {
			VecOut[neuron] = sum + biases[neuronAbs];
		}
		else if (extraParams[0] == 2) {
			float z = sum + biases[neuronAbs];
			VecOut[neuron] = 1 / (1 + exp(-z));
		}
		else if (extraParams[0] == 3) {
			float z = sum + biases[neuronAbs];
			VecOut[neuron] = z * (1 / (1 + exp(-z)));
		}
	}
}

std::vector<std::vector<float>> NeuralNetwork::forwardPassGPU(const std::vector<std::vector<float>>& input, size_t numberOfStreams) const
{

	for (auto in : input) {
		if (m_top[0] != in.size()) {
			std::stringstream ss;
			ss << "input layer is not the same size as the parameters passed: " << m_top[0] << " vs " << input.size() << std::endl;
			throw std::invalid_argument(ss.str());
		}
	}



	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}



	float* inputGPU = 0;
	float* outputFromLayer = 0;

	unsigned int inputLayerSize = input[0].size();

	size_t initialIntermediateInstanceSize = m_intermediateAValues.size();

	for (size_t i = 0; i < m_hiddenLayers.size() + 1; i++) {


		const Layer* layer;
		if (i < m_hiddenLayers.size()) {
			layer = &m_hiddenLayers[i];
		}
		else {
			layer = &m_outputLayer;
		}

			
		if (i == 0) {

			CUDA_CHECK(cudaMalloc((void**)&inputGPU, input[0].size()*input.size() * sizeof(float)));

			for (size_t o = 0; o < input.size(); o++) {
				CUDA_CHECK(cudaMemcpy(inputGPU+(input[o].size()*o), input[o].data(), input[o].size() * sizeof(float), cudaMemcpyHostToDevice));
			}


			//start recording a new  instance of intermediate values, first entry in the instance is the input.
			if (m_recording) {
				for (size_t o = 0; o < input.size(); o++) {
					m_intermediateAValues.push_back({});
					m_intermediateZValues.push_back({});
					m_instanceN++;
					float* zValues = 0;
					cudaStatus = cudaMalloc((void**)&zValues, input[o].size() * sizeof(float));
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMalloc failed!");
					}


					float* aValues = 0;
					cudaStatus = cudaMalloc((void**)&aValues, input[o].size() * sizeof(float));
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMalloc failed!");
					}

					cudaStatus = cudaMemcpy(zValues, input[o].data(), input[o].size() * sizeof(float), cudaMemcpyHostToDevice);
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMemcpy failed!");
					}

					cudaStatus = cudaMemcpy(aValues, input[o].data(), input[o].size() * sizeof(float), cudaMemcpyHostToDevice);
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMemcpy failed!");
					}

					m_intermediateAValues.back().push_back(aValues);
					m_intermediateZValues.back().push_back(zValues);
				}
			}
		}


		float* zValues = 0;

		cudaStatus = cudaMalloc((void**)&zValues, input.size()*layer->size * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}



		cudaStatus = cudaMalloc((void**)&outputFromLayer, input.size()*layer->size * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}




		unsigned int blockSizeX = std::max((unsigned int)std::ceilf(layer->size/THREADS_PER_BLOCK), (unsigned int)1);
		unsigned int threadSizeX = std::min((size_t)THREADS_PER_BLOCK,layer->size);

		dim3 blockDim(blockSizeX,input.size(),1);


		forwardPassLayerKernel << < blockDim, threadSizeX >> > (zValues, outputFromLayer, inputGPU, std::get<0>(m_GPULayers[i]), std::get<1>(m_GPULayers[i]), std::get<2>(m_GPULayers[i]), layer->size, inputLayerSize);

		CUDA_CHECK(cudaGetLastError());

		// Check for any errors launching the kernel


		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching processLayerKernel!\n", cudaStatus);
		}




		if (m_recording && i < m_hiddenLayers.size()) {
			for (size_t o = 0; o < input.size(); o++) {
				float* sliceOutputFromLayer = 0;
				CUDA_CHECK(cudaMalloc((void**)&sliceOutputFromLayer, layer->size*sizeof(float)));
				CUDA_CHECK(cudaMemcpy(sliceOutputFromLayer, outputFromLayer + (o * layer->size), layer->size* sizeof(float),cudaMemcpyDeviceToDevice));
				m_intermediateAValues[o + initialIntermediateInstanceSize].push_back(sliceOutputFromLayer);
				float* sliceZValues = 0;
				CUDA_CHECK(cudaMalloc((void**)&sliceZValues, layer->size* sizeof(float)));
				CUDA_CHECK(cudaMemcpy(sliceZValues, zValues + (o * layer->size), layer->size* sizeof(float), cudaMemcpyDeviceToDevice));
				m_intermediateZValues[o + initialIntermediateInstanceSize].push_back(sliceZValues);
			}
		}







		inputLayerSize = layer->size;

		std::vector<std::vector<float>> retval(input.size());

		if (i >= m_hiddenLayers.size()) {
			CUDA_CHECK(cudaFree(inputGPU));
			float* out = new float[input.size()*layer->size];
			// Copy output vector from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(out, outputFromLayer, input.size()*layer->size * sizeof(float), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}

			for (size_t o = 0; o < input.size(); o++) {
				retval[o].insert(retval[o].begin(), out+(layer->size*o), out + (layer->size*(o+1)));
			}

			

			delete out;

		}
		else {
			CUDA_CHECK(cudaFree(inputGPU));

			cudaStatus = cudaMalloc((void**)&inputGPU, input.size()*layer->size * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
			}
			cudaStatus = cudaMemcpy(inputGPU, outputFromLayer, input.size()*layer->size * sizeof(float), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		CUDA_CHECK(cudaFree(zValues));

		CUDA_CHECK(cudaFree(outputFromLayer));

		if (i >= m_hiddenLayers.size()) {

			return retval;
		}


	}

	return std::vector<std::vector<float>>();
}




std::string NeuralNetwork::serialize()
{
	m_updateRAM();
	std::stringstream ss;
	ss << m_epoch << " " << std::endl;
	for (auto x : m_top)
	{
		ss << x << " ";
	}
	ss << std::endl;


	for (size_t i = 0; i < m_hiddenLayers.size()+1; i++)
	{
		const Layer *layer;
		if (i < m_hiddenLayers.size()) {
			layer = &m_hiddenLayers[i];
		}
		else {
			layer = &m_outputLayer;
		}

		for (size_t o = 0; o < layer->size; o++) {
			ss << layer->biases[o] << " ";
			auto &synapses = layer->weights[o];
			for (auto syn : synapses) {
				ss << syn << " ";
			}
			ss << "/";
		}
		ss << ",";
	}
	return ss.str();
}

NeuralNetwork::~NeuralNetwork()
{
	for (auto &GPULayer : m_GPULayers) {
		cudaFree(std::get<0>(GPULayer));
		cudaFree(std::get<1>(GPULayer));
		cudaFree(std::get<2>(GPULayer));
	}
}

void NNInitialization::seedEngine()
{
	engine.seed();
}

float NNInitialization::generateRandomNumber(float fan)
{
	std::uniform_real_distribution<float> dist;

	float r;
	switch (initType)
	{
	case NNInitialization::He:
		r = sqrt(6 / fan);
		dist = std::uniform_real_distribution<float>(-r,r);
		break;
	default:
		r = 1;
		dist = std::uniform_real_distribution<float>(-r, r);
		break;
	}

	return dist(NNInitialization::engine);
}


void Layer::randomizeParameters(LearningSchedule& ls)
{
	for (size_t i = 0; i < size; i++) {
		weights.push_back(std::vector<float>());
		for (size_t o = 0; o < prevSize; o++) {
			weights[i][o] += ls.generateRandomNumber();
		}
	}
	if (!isOutput) {
		for (size_t i = 0; i < size; i++) {
			biases[i] += ls.generateRandomNumber();
		}
	}
}

Layer::Layer(NNInitialization& init, size_t p_size, size_t p_prevSize, bool p_isOutput, NNActivation p_act):
	size(p_size),
	prevSize(p_prevSize),
	act(p_act),
	isOutput(p_isOutput)
{

	for (size_t i = 0; i < size; i++) {
		weights.push_back(std::vector<float>());
		for (size_t o = 0; o < prevSize; o++) {
			weights[i].push_back(init.generateRandomNumber(p_prevSize));
		}
	}
	if (isOutput) {
		biases = std::vector<float>(size, 0.0f);
		act = NNActivation(activationType::linear);
	}
	else {
		for (size_t i = 0; i < size; i++) {
			biases.push_back(init.generateRandomNumber(p_prevSize));
		}
	}
}

Layer::Layer():
	act(NNActivation(activationType::sigmoid))
{
}

NNActivation::NNActivation(activationType p_actType)
{
	if (p_actType == activationType::other) {
		throw std::invalid_argument("Must be activation type that is not other");
	}
	actType = p_actType;
	switch (p_actType) {
	case activationType::sigmoid:
		func = [](float in) {return 1 / (1 + exp(-in));};
		break;
	case activationType::linear:
		func = [](float in) {return in;};
		break;
	case activationType::swish:
		func = [](float in) {return in*(1 / (1 + exp(-in)));};
	}
}

NNActivation::NNActivation(activationFunc p_func)
{
	actType = activationType::other;
	func = p_func;
}


float LearningSchedule::getLearningRate(size_t epoch)
{
	if (constantLearningRate) {
		return learningRate;
	}
	float warmupLRate = initialLearningRate + epoch * warmupIncrementPerEpoch;
	float retval;
	if (useLinearDecrease) {
		retval = std::max(initialLearningRate, learningRate-epoch*linearDecreasePerEpoch);
	} else{
		retval = learningRate * pow(0.1, epoch / learningRateDecreaseFactor);
	}


	if (useWarmup && warmupLRate < learningRate) {
		return warmupLRate;
	}

	if (useWarmup && !useLinearDecrease) {
		return learningRate * pow(0.1, (epoch - ((learningRate - initialLearningRate) / warmupIncrementPerEpoch)) / learningRateDecreaseFactor);
	}

	if (useLinearDecrease && useWarmup) {
		return std::max(initialLearningRate, learningRate - (epoch - ((learningRate - initialLearningRate) / warmupIncrementPerEpoch)) * linearDecreasePerEpoch);
	}

	return retval;

}

float LearningSchedule::generateRandomNumber()
{
	return dist(NNInitialization::engine);
}
