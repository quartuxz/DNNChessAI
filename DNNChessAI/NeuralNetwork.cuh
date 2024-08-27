#pragma once
#include <vector>
#include <random>
#include <mutex>
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


struct LearningSchedule {
	float learningRate =            0.01;
	float initialLearningRate =     0.001;
	float warmupIncrementPerEpoch = 0.001;
	float linearDecreasePerEpoch =  0.00005;
	bool useWarmup = true;
	bool useLinearDecrease = false;
	bool constantLearningRate = false;
	float learningRateDecreaseFactor = 20;
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

	std::vector<std::vector<float>> m_accumulatedInstancesForBP;


	Topology m_top;

	NNActivation m_act;

	size_t m_epoch = 0;

	std::vector<GPUDataLayer> m_GPULayers;
	//contains the momentum of each parameter(first weights and second biases) in each layer.
	std::vector<std::pair<float*, float*>> m_GPUMomentum;

	void m_updateGPUMem();

	void m_updateRAM();

	void m_initializeGpuMem();

public:
	LearningSchedule m_learningSched;


	NeuralNetwork(Topology top, NNInitialization init, LearningSchedule ls ,NNActivation act = NNActivation(activationType::swish));
	//deserializing function
	NeuralNetwork(const std::string& str);

	size_t accumulateInstanceForBackprop(const std::vector<std::vector<float>> &instaceOutputDif);

	void addRandomWeights();

	NeuralNetwork(NeuralNetwork&);
	friend void swap(NeuralNetwork& first, NeuralNetwork& second) {
		first.m_updateRAM();
		second.m_updateRAM();
		std::swap(first.m_GPULayers, second.m_GPULayers);
		std::swap(first.m_top, second.m_top);
		std::swap(first.m_learningSched,second.m_learningSched);
		std::swap(first.m_act, second.m_act);
		std::swap(first.m_hiddenLayers, second.m_hiddenLayers);
		std::swap(first.m_outputLayer, second.m_outputLayer);
		std::swap(first.m_batchN, second.m_batchN);
	}
	NeuralNetwork &operator=(NeuralNetwork);


	void increaseEpoch();
	void setEpoch(size_t epoch);
	size_t getEpoch()const;

	//numberOfStreams is the synchronization upper bound not the exact value
	void backpropagateGPU(std::vector<std::vector<float>> &dCost_dOutput_forInstances, size_t numberOfStreams = 64);
	//uses saved values for each instance
	void backpropagateGPU(size_t numberOfStreams = 64);
	void startRecording();
	void endRecording();
	//select a specific training instance recorded between startRecording and endRecording to be saved.
	void selectAndDiscardRest(unsigned int selected, bool selectAll = false);
	void clearTrainingData();

	//only works when testing, cannot save training values.
	std::vector<float> forwardPassCPU(const std::vector<float> &input);
	//receives a batch of inputs, passes them through the entire neural net, and gets a batch of outputs.
	//numberOfStreams is the synchronization upper bound not the exact value
	std::vector<std::vector<float>> forwardPassGPU(const std::vector<std::vector<float>> &input, size_t numberOfStreams = 128)const;


	std::string serialize();

	~NeuralNetwork();

};