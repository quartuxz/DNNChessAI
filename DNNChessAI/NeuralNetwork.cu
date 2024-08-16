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


Topology getTopology(const std::string& str) {
	std::string currentNumber;

	Topology retval;

	for (char current : str) {
		if (current == '\n') {
			break;
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
			if (line >= 1) {
				break;
			}
			line++;
			continue;
		}
		if (line < 1) {
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

NeuralNetwork::NeuralNetwork(const NeuralNetwork &other):
	m_outputLayer(other.m_outputLayer),
	m_act(other.m_act),
	m_hiddenLayers(other.m_hiddenLayers),
	m_learningSched(other.m_learningSched),
	m_top(other.m_top),
	m_batchN(other.m_batchN)
{
	m_initializeGpuMem();
	m_updateGPUMem();
}

NeuralNetwork& NeuralNetwork::operator=(NeuralNetwork other)
{
	swap(*this,other);
	return *this;
}

__global__ void calculateZErrorKernel(float *thisZError, float *thisBiasAcc, const float *thisZ,const float *thisA, const float *nextZError,  const float *weightMatrix, const unsigned int* extraParamsThisLayer, const unsigned int *extraParamsNextLayer)
{
	int neuron = threadIdx.x + blockDim.x * blockIdx.x;
	if (neuron < extraParamsThisLayer[2]) {
		const unsigned int &matrixWidth = extraParamsThisLayer[2];
		const unsigned int &matrixHeight = extraParamsNextLayer[2];
		float errorWeightSum = 0;
		for (size_t i = 0; i < matrixHeight; i++) {
			errorWeightSum += weightMatrix[neuron + matrixWidth * i]*nextZError[i];
		}
		float deriv = 0;
		if (extraParamsThisLayer[0] == 2) {

			deriv = thisA[neuron] * (1 - thisA[neuron]);
		}
		else if (extraParamsThisLayer[0] == 3) {
			float sigmoid = 1 / (1 + exp(-thisZ[neuron]));
			deriv = thisA[neuron]+sigmoid*(1-thisA[neuron]);
		}


		thisZError[neuron] = errorWeightSum * deriv;

		thisBiasAcc[neuron] += thisZError[neuron];
	}
}

__global__ void calculateWeightGradientsKernel(float *weightMatrixAcc, const float *thisZError, const float *prevA, const unsigned int *extraParams) {
	int neuron = threadIdx.x + blockDim.x * blockIdx.x;
	if (neuron < extraParams[2]) {
		for (size_t i = 0; i < extraParams[1]; i++) {
			weightMatrixAcc[i + neuron * extraParams[1]] += prevA[i]*thisZError[neuron];
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
			weights[i + neuron * extraParams[1]] += momentumWeights[i + neuron * extraParams[1]];
		}
		momentumBiases[neuron] = instancesAndLearningRate[2] * momentumBiases[neuron] - instancesAndLearningRate[1] * (biasesAcc[neuron] / instancesAndLearningRate[0]);
		biases[neuron] += momentumBiases[neuron];
	}
}


void NeuralNetwork::backpropagateGPU(std::vector<std::vector<float>>& dCost_dOutput_forInstances, size_t epoch, size_t numberOfStreams)
{

	m_instancesInBatch = 0;
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



	std::vector<cudaStream_t> streams(dCost_dOutput_forInstances.size());
	
	//first axis is layer second axis is instance
	std::vector<std::vector<float*>> dCost_dErrorL;
	dCost_dErrorL.push_back({});

	//reverse order, last layer first
	for (int o = m_hiddenLayers.size() - 1; o >= 0; o--) {
		dCost_dErrorL.push_back({});
		for (size_t i = 0; i < dCost_dOutput_forInstances.size();i++) {




			if (o == m_hiddenLayers.size() - 1) {
				//initialize streams
				cudaStreamCreate(&streams[i]);

				//initialize output layer errors
				dCost_dErrorL[0].push_back(0);
				cudaStatus = cudaMalloc((void**)&dCost_dErrorL[0].back(), dCost_dOutput_forInstances[i].size() * sizeof(float));
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMalloc failed!, dCost_dErrorL, backpropagateGPU %s\n", cudaGetErrorString(cudaStatus));
					system("pause");
				}
				cudaStatus = cudaMemcpy(dCost_dErrorL[0].back(), dCost_dOutput_forInstances[i].data(), dCost_dOutput_forInstances[i].size() * sizeof(float), cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy failed!, dCost_dErrorL, backpropagateGPU");
					system("pause");
				}
			}

			//error in the next layer for the current instance
			float* nextZError = dCost_dErrorL[(m_hiddenLayers.size() - 1)-o][i];
			dCost_dErrorL.back().push_back(0);
			cudaStatus = cudaMalloc((void**)&dCost_dErrorL.back().back(), m_hiddenLayers[o].size * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!, dCost_dErrorL, loop 2, backpropagateGPU, %s\n", cudaGetErrorString(cudaStatus));
				system("pause");
			}


			unsigned int blockSize = std::max((unsigned int)std::ceilf(m_hiddenLayers[o].size / THREADS_PER_BLOCK), (unsigned int)1);
			unsigned int threadSize = std::min((size_t)THREADS_PER_BLOCK, m_hiddenLayers[o].size);


			// Launch a kernel on the GPU with one thread for each element.
			//accumulate biases for this instance
			calculateZErrorKernel <<< blockSize, threadSize, 0, streams[i] >>> (dCost_dErrorL.back().back(), biasAcc[o], m_savedZValues[i][o + 1], m_savedAValues[i][o + 1], nextZError, std::get<0>(m_GPULayers[o + 1]), std::get<2>(m_GPULayers[o]), std::get<2>(m_GPULayers[o + 1]));

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "calculateZErrorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				system("pause");
			}

			//accumulate weights for this instance
			calculateWeightGradientsKernel <<< blockSize, threadSize, 0, streams[i] >>> (weightAcc[o], dCost_dErrorL.back().back(), m_savedAValues[i][o], std::get<2>(m_GPULayers[o]));

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "calculateWeightGradientsKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				system("pause");
			}


			if (i%numberOfStreams == numberOfStreams -1) {
				// cudaDeviceSynchronize waits for the kernel to finish, and returns
				// any errors encountered during the launch.
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculateWeightGradientsKernel and other!\n", cudaStatus);
					system("pause");
				}
			}
		}
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculateWeightGradientsKernel and other!\n", cudaStatus);
			system("pause");
		}

		for (auto instance: dCost_dErrorL[(m_hiddenLayers.size() - 1) - o]) {
			cudaFree(instance);
		}
	}

	for (auto stream : streams) {
		cudaStreamDestroy(stream);
	}

	streams.clear();

	for (auto instance : dCost_dErrorL.back()) {
		cudaFree(instance);
	}
	dCost_dErrorL.clear();

	float* instancesAndLearningRate = new float[3];
	instancesAndLearningRate[0] = dCost_dOutput_forInstances.size();
	instancesAndLearningRate[1] = m_learningSched.getLearningRate(epoch);
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

	cudaFree(instancesAndLearningRateGPU);



	m_updateRAM();

	for (auto weight : weightAcc) {
		cudaFree(weight);
	}
	for (auto bias : biasAcc) {
		cudaFree(bias);
	}

	clearTrainingData();
	m_batchN++;
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
					cudaFree(val);
				}
			}
		}
		for (size_t i = 0; i < m_intermediateZValues.size(); i++) {
			if (i != selected) {
				for (auto val : m_intermediateZValues[i]) {
					cudaFree(val);
				}
			}
		}
	}



	m_intermediateAValues.clear();
	m_intermediateZValues.clear();

}

void NeuralNetwork::clearTrainingData()
{
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

std::vector<float> NeuralNetwork::forwardPassCPU(const std::vector<float>& input) const
{
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




__global__ void forwardPassLayerKernel(float* zValues, float* VecOut, const float* VecIn, const float* weights, const float* biases, const unsigned int* extraParams)
{
	int neuron = threadIdx.x + blockDim.x * blockIdx.x;
	if (neuron < extraParams[2]) {

		float sum = 0;
		for (size_t i = 0; i < extraParams[1]; i++) {
			sum += weights[i + neuron * extraParams[1]] * VecIn[i];
		}
		zValues[neuron] = sum+biases[neuron];
		if (extraParams[0] == 1) {
			VecOut[neuron] = sum + biases[neuron];
		}
		else if (extraParams[0] == 2) {
			float z = sum + biases[neuron];
			VecOut[neuron] = 1 / (1 + exp(-z));
		}
		else if(extraParams[0] == 3) {
			float z = sum + biases[neuron];
			VecOut[neuron] = z * (1 / (1 + exp(-z)));
		}
	}
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



std::vector<std::vector<float>> NeuralNetwork::forwardPassGPU(const std::vector<std::vector<float>>& input, size_t numberOfStreams) const
{


	//for (auto in : input) {
	//	if (m_top[0] != in.size()) {
	//		std::stringstream ss;
	//		ss << "input layer is not the same size as the parameters passed: " << m_top[0] << " vs " << input.size() << std::endl;
	//		throw std::invalid_argument(ss.str());
	//	}
	//}

	//cudaError_t cudaStatus;

	//// Choose which GPU to run on, change this on a multi-GPU system.
	//cudaStatus = cudaSetDevice(0);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	//}
	//std::vector<std::vector<float>> retval(input.size());

	//for (size_t o = 0; o < input.size(); o++) {
	//	float* inputGPU = 0;
	//	cudaStatus = cudaMalloc((void**)&inputGPU, input[o].size() * sizeof(float));
	//	if (cudaStatus != cudaSuccess) {
	//		fprintf(stderr, "cudaMalloc failed!");
	//	}


	//	cudaStatus = cudaMemcpy(inputGPU, input[o].data(), input[o].size() * sizeof(float), cudaMemcpyHostToDevice);
	//	if (cudaStatus != cudaSuccess) {
	//		fprintf(stderr, "cudaMemcpy failed!");
	//	}

	//	//start recording a new  instance of intermediate values, first entry in  the instance is the input.
	//	if (m_recording) {
	//		m_intermediateAValues.push_back({});
	//		m_intermediateZValues.push_back({});
	//		m_instanceN++;
	//		float* zValues = 0;
	//		cudaStatus = cudaMalloc((void**)&zValues, input[o].size() * sizeof(float));
	//		if (cudaStatus != cudaSuccess) {
	//			fprintf(stderr, "cudaMalloc failed!");
	//		}


	//		float* aValues = 0;
	//		cudaStatus = cudaMalloc((void**)&aValues, input[o].size() * sizeof(float));
	//		if (cudaStatus != cudaSuccess) {
	//			fprintf(stderr, "cudaMalloc failed!");
	//		}

	//		cudaStatus = cudaMemcpy(zValues, input[o].data(), input[o].size() * sizeof(float), cudaMemcpyHostToDevice);
	//		if (cudaStatus != cudaSuccess) {
	//			fprintf(stderr, "cudaMemcpy failed!");
	//		}

	//		cudaStatus = cudaMemcpy(aValues, input[o].data(), input[o].size() * sizeof(float), cudaMemcpyHostToDevice);
	//		if (cudaStatus != cudaSuccess) {
	//			fprintf(stderr, "cudaMemcpy failed!");
	//		}

	//		m_intermediateAValues.back().push_back(aValues);
	//		m_intermediateZValues.back().push_back(zValues);
	//	}



	//	for (size_t i = 0; i < m_hiddenLayers.size() + 1; i++) {

	//		const Layer* layer;
	//		if (i < m_hiddenLayers.size()) {
	//			layer = &m_hiddenLayers[i];
	//		}
	//		else {
	//			layer = &m_outputLayer;
	//		}

	//		float* zValues = 0;

	//		cudaStatus = cudaMalloc((void**)&zValues, layer->size * sizeof(float));
	//		if (cudaStatus != cudaSuccess) {
	//			fprintf(stderr, "cudaMalloc failed!");
	//		}

	//		float* output = 0;

	//		cudaStatus = cudaMalloc((void**)&output, layer->size * sizeof(float));
	//		if (cudaStatus != cudaSuccess) {
	//			fprintf(stderr, "cudaMalloc failed!");
	//		}



	//		unsigned int blockSize = std::max((unsigned int)std::ceilf(layer->size / THREADS_PER_BLOCK), (unsigned int)1);
	//		unsigned int threadSize = std::min((size_t)THREADS_PER_BLOCK, layer->size);
	//		forwardPassLayerKernel << < blockSize, threadSize >> > (zValues, output, inputGPU, std::get<0>(m_GPULayers[i]), std::get<1>(m_GPULayers[i]), std::get<2>(m_GPULayers[i]));

	//		if (m_recording) {
	//			m_intermediateAValues.back().push_back(output);
	//			m_intermediateZValues.back().push_back(zValues);
	//		}
	//		else {
	//			cudaFree(zValues);
	//		}


	//		// Check for any errors launching the kernel
	//		cudaStatus = cudaGetLastError();
	//		if (cudaStatus != cudaSuccess) {
	//			fprintf(stderr, "processLayerKernel launch failed: %s%s\n", cudaGetErrorString(cudaStatus));
	//			system("pause");
	//		}

	//		// cudaDeviceSynchronize waits for the kernel to finish, and returns
	//		// any errors encountered during the launch.
	//		cudaStatus = cudaDeviceSynchronize();
	//		if (cudaStatus != cudaSuccess) {
	//			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching processLayerKernel!\n", cudaStatus);
	//		}




	//		if (i == m_hiddenLayers.size()) {
	//			cudaFree(inputGPU);
	//			float* out = new float[layer->size];
	//			// Copy output vector from GPU buffer to host memory.
	//			cudaStatus = cudaMemcpy(out, output, layer->size * sizeof(float), cudaMemcpyDeviceToHost);
	//			if (cudaStatus != cudaSuccess) {
	//				fprintf(stderr, "cudaMemcpy failed!");
	//			}
	//			if (!m_recording) {
	//				cudaFree(output);
	//			}


	//			retval[o] = std::vector<float>(out, out + layer->size);
	//			delete out;
	//			break;
	//		}
	//		else {
	//			cudaFree(inputGPU);

	//			cudaStatus = cudaMalloc((void**)&inputGPU, layer->size * sizeof(float));
	//			if (cudaStatus != cudaSuccess) {
	//				fprintf(stderr, "cudaMalloc failed!");
	//			}
	//			cudaStatus = cudaMemcpy(inputGPU, output, layer->size * sizeof(float), cudaMemcpyDeviceToDevice);
	//			if (cudaStatus != cudaSuccess) {
	//				fprintf(stderr, "cudaMemcpy failed!");
	//			}
	//		}

	//		cudaStatus = cudaDeviceSynchronize();
	//		if (cudaStatus != cudaSuccess) {
	//			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching processLayerKernel!\n", cudaStatus);
	//		}

	//		if (!m_recording) {
	//			cudaStatus = cudaFree(output);
	//			if (cudaStatus != cudaSuccess) {
	//				fprintf(stderr, "cudaFree failed!");
	//			}
	//		}
	//	}
	//}
	//
	//return retval;


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



	std::vector<float*> inputGPU(input.size());
	std::vector<float*> output(input.size());

	std::vector<cudaStream_t> streams(input.size());


	for (size_t i = 0; i < m_hiddenLayers.size() + 1; i++) {
		const Layer* layer;
		if (i < m_hiddenLayers.size()) {
			layer = &m_hiddenLayers[i];
		}
		else {
			layer = &m_outputLayer;
		}


		for (size_t o = 0; o < input.size(); o++) {

			if (i == 0) {

				cudaStreamCreate(&streams[o]);

				cudaStatus = cudaMalloc((void**)&inputGPU[o], input[o].size() * sizeof(float));
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMalloc failed!");
				}


				cudaStatus = cudaMemcpy(inputGPU[o], input[o].data(), input[o].size() * sizeof(float), cudaMemcpyHostToDevice);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy failed!");
				}

				//start recording a new  instance of intermediate values, first entry in the instance is the input.
				if (m_recording) {
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


			float* zValues = 0;

			cudaStatus = cudaMalloc((void**)&zValues, layer->size * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
			}



			cudaStatus = cudaMalloc((void**)&output[o], layer->size * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
			}



			unsigned int blockSize = std::max((unsigned int)std::ceilf(layer->size/THREADS_PER_BLOCK), (unsigned int)1);
			unsigned int threadSize = std::min((size_t)THREADS_PER_BLOCK,layer->size);
			forwardPassLayerKernel<<< blockSize, threadSize,0,streams[o] >> >(zValues, output[o], inputGPU[o], std::get<0>(m_GPULayers[i]), std::get<1>(m_GPULayers[i]), std::get<2>(m_GPULayers[i]));

			if (m_recording && i < m_hiddenLayers.size()) {
				m_intermediateAValues[o].push_back(output[o]);
				m_intermediateZValues[o].push_back(zValues);
			}
			else {
				cudaFree(zValues);
			}


			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "processLayerKernel launch failed: %s%s\n", cudaGetErrorString(cudaStatus));
				system("pause");
			}

			if (o%numberOfStreams== numberOfStreams-1) {
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching processLayerKernel!\n", cudaStatus);
				}
			}
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching processLayerKernel!\n", cudaStatus);
		}

		std::vector<std::vector<float>> retval(input.size());
		for (size_t o = 0; o < input.size(); o++) {
			if (i >= m_hiddenLayers.size()) {
				cudaFree(inputGPU[o]);
				float* out = new float[layer->size];
				// Copy output vector from GPU buffer to host memory.
				cudaStatus = cudaMemcpy(out, output[o], layer->size * sizeof(float), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy failed!");
				}

				retval[o].insert(retval[o].begin(),out,out+layer->size);

				if (m_recording) {
					cudaStatus = cudaFree(output[o]);
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaFree failed!");
					}
				}

				delete out;

			}
			else {
				cudaFree(inputGPU[o]);

				cudaStatus = cudaMalloc((void**)&inputGPU[o], layer->size * sizeof(float));
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMalloc failed!");
				}
				cudaStatus = cudaMemcpy(inputGPU[o], output[o], layer->size * sizeof(float), cudaMemcpyDeviceToDevice);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy failed!");
				}
			}
			if (!m_recording) {
				cudaStatus = cudaFree(output[o]);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaFree failed!");
				}
			}
		}
		if (i >= m_hiddenLayers.size()) {

			for (auto stream: streams) {
				cudaStreamDestroy(stream);
			}

			return retval;
		}
	}

	return std::vector<std::vector<float>>();
}



std::string NeuralNetwork::serialize()const
{

	std::stringstream ss;
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
	return learningRate * pow(0.1,epoch/20);
}

float LearningSchedule::generateRandomNumber()
{
	return dist(NNInitialization::engine);
}
