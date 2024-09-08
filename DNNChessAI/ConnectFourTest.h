#pragma once
#include <vector>
#include "NeuralNetwork.cuh"
#include "SamplingVector.h"

#define C4_DEFAULT_TOPOLOGY {3*7*6,200,200,200,200,200,200,200,200,1}

typedef std::vector<std::vector<int>> boardC4_t;



enum class playerC4 :
	unsigned int
{
	none = 0,
	first = 1,
	second = 2
};


struct connectFourPlay {
	boardC4_t boardAfterPlay;
	playerC4 whoPlayed;
	float reward = 0;
	bool operator<(const connectFourPlay& other);
};



class ConnectFourTest
{
private:
	struct C4PlayOGCandidate {
		connectFourPlay play;
		size_t originalCandidate;
		bool operator<(const C4PlayOGCandidate& other);
		bool operator>(const C4PlayOGCandidate& other)const;
	};
	NeuralNetwork *m_nn;
	NeuralNetwork* m_targetNN  = nullptr;
	

	SamplingVector<connectFourPlay> m_replayBuffer = SamplingVector<connectFourPlay>(0.6,1000000);

	bool m_tracking = true;

	float m_epsilon = 1;
	float m_betaWeight = 0.4;

	float m_discount = 0.94;

	size_t m_targetLag = 100;
	size_t m_batchSize = 64;
	size_t m_subpasses = 2500;

	std::string m_savefile = "nn_v1";

	void m_generateTrainingData(float difficulty, size_t amount);


	void m_minimaxNN(std::pair<boardC4_t*, playerC4> game, size_t depth, size_t turrns);

	void m_makeMoveWithDepthAndBreadth(std::pair<boardC4_t*, playerC4> game, size_t depth = 1, size_t breadth = 1, bool highestTrueSampleFalse = true);


	void m_trainOnce();
	
public:

	void setSaveFile(const std::string& name);

	std::vector<float> m_makeMovesWithNN(std::vector<std::pair<boardC4_t*, playerC4>> games, NeuralNetwork *nn = nullptr,std::vector<std::vector<connectFourPlay> > *lines = nullptr, bool highestTrueSampleFalse = true, bool backpropagationTraining = false, size_t ffBatchSize = 2048);

	ConnectFourTest(Topology top = C4_DEFAULT_TOPOLOGY);

	ConnectFourTest(std::string fileName);

	void train(size_t episodes, size_t afterTraining);
	std::string test(size_t games);
	void save();
	void play();
	~ConnectFourTest();
};

