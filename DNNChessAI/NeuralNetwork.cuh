#pragma once
#include <vector>
#include <random>
#include <mutex>
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include "cuda_runtime.h"



enum class activationType : uint8_t {
	other = 0,
	linear = 1,
	sigmoid = 2,
	swish = 3

};

typedef float (*activationFunc)(float);

struct NNActivation {
	activationFunc func;
	activationType actType;
	NNActivation(activationType p_actType);
	NNActivation(activationFunc p_func);
};



//also takes care of initialization
class NNInitialization {
public:
	static std::default_random_engine engine;
	static void seedEngine();
	
	enum initType_t {
		He
	}initType;


	float generateRandomNumber(float fan);
};

//for mutations
struct LearningSchedule {
	float learningRate = 0.01;
	float momentum = 0.9;
	std::uniform_real_distribution<float> dist = std::uniform_real_distribution<float>(-learningRate,learningRate);
	float getLearningRate(size_t epoch = 0);
	float generateRandomNumber();
};


typedef std::vector<size_t> Topology;


struct Layer {
	//first index is the receiving neuron, second is the index of the providing neuron.
	std::vector<std::vector<float>> weights;
	std::vector<float> biases;
	NNActivation act;
	bool isOutput;
	size_t size;
	size_t prevSize;

	void randomizeParameters(LearningSchedule &ls);
	Layer(NNInitialization& init, size_t p_size, size_t p_incoming, bool p_isOutput = false, NNActivation p_act = NNActivation(activationType::swish));
	Layer();
};

//contains all the weights for a layer in a 1d array, biases in the next array and extra information like activation function in the size_t*
typedef std::tuple<float*,float*, unsigned int*> GPUDataLayer;

class NeuralNetwork
{
private:
	std::vector<Layer> m_hiddenLayers;
	Layer m_outputLayer;
	

	//used for backpropagation
	std::vector<std::vector<float*>> m_savedZValues;
	std::vector<std::vector<float*>> m_savedAValues;
	mutable std::vector<std::vector<float*>>  m_intermediateZValues;
	mutable std::vector<std::vector<float*>> m_intermediateAValues;
	mutable size_t m_instanceN = 0;
	mutable size_t m_instancesInBatch = 0;
	size_t m_batchN = 0;
	bool m_recording = false;


	Topology m_top;
	LearningSchedule m_learningSched;
	NNActivation m_act;

	std::vector<GPUDataLayer> m_GPULayers;
	//contains the momentum of each parameter(first weights and second biases) in each layer.
	std::vector<std::pair<float*, float*>> m_GPUMomentum;

	void m_updateGPUMem();

	void m_updateRAM();

	void m_initializeGpuMem();

public:

	NeuralNetwork(Topology top, NNInitialization init, LearningSchedule ls ,NNActivation act = NNActivation(activationType::swish));
	//deserializing function
	NeuralNetwork(const std::string& str);

	void addRandomWeights();

	NeuralNetwork(const NeuralNetwork&);
	friend void swap(NeuralNetwork& first, NeuralNetwork& second) {
		std::swap(first.m_GPULayers, second.m_GPULayers);
		std::swap(first.m_top, second.m_top);
		std::swap(first.m_learningSched,second.m_learningSched);
		std::swap(first.m_act, second.m_act);
		std::swap(first.m_hiddenLayers, second.m_hiddenLayers);
		std::swap(first.m_outputLayer, second.m_outputLayer);
		std::swap(first.m_batchN, second.m_batchN);
	}
	NeuralNetwork &operator=(NeuralNetwork);


	void backpropagateGPU(std::vector<std::vector<float>> &dCost_dOutput_forInstances, size_t epoch = 0,size_t numberOfStreams = 1);
	void startRecording();
	void endRecording();
	//select a specific training instance recorded between startRecording and endRecording to be saved.
	void selectAndDiscardRest(unsigned int selected, bool selectAll = false);
	void clearTrainingData();

	//only works when testing, cannot save training values.
	std::vector<float> forwardPassCPU(const std::vector<float> &input)const;
	//receives a batch of inputs, passes them through the entire neural net, and gets a batch of outputs.
	std::vector<std::vector<float>> forwardPassGPU(const std::vector<std::vector<float>> &input, size_t numberOfStreams = 1)const;


	std::string serialize()const;

	~NeuralNetwork();

};