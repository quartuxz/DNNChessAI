#pragma once
#include <vector>
#include "NeuralNetwork.cuh"

#define C4_DEFAULT_TOPOLOGY {3*7*6,200,200,200,200,200,200,200,200,200,200,200,1}

typedef std::vector<std::vector<int>> boardC4_t;



enum class playerC4 :
	unsigned int
{
	none = 0,
	first = 1,
	second = 2
};

class ConnectFourTest
{
private:
	std::vector<NeuralNetwork> m_ensemble;



public:
	boardC4_t m_predictWithEnsemble(boardC4_t board, playerC4 p);
	ConnectFourTest(Topology top, size_t players);

	ConnectFourTest(std::vector<std::string> fileNames);

	void train(size_t epochs, size_t games);
	std::string test(size_t games);
	void save(std::string nameConv);
	void play();
};

