#pragma once

#include <tuple>
#include <thread>
#include <queue>
#include <sstream>
#include "ChessGame.cuh"
#include "NeuralNetwork.cuh"
#include "RealGameSampler.cuh"

#define QEAC_DEFAULT_TOPOLOGY {838,8192,8192,8192,8192,8192,8192,1}
//#define QEAC_DEFAULT_TOPOLOGY {838,8192,8192,1}

struct Match {
	size_t black;
	size_t white;
	ChessGame gameState;
};

gameCondition matchTwoNNs(ChessGame &game, NeuralNetwork* black, NeuralNetwork* white, std::vector < std::pair<float,unsigned int> >* probasWhite = nullptr, std::vector < std::pair<float, unsigned int> >* probasBlack = nullptr);


gameCondition makeMoveWithNN(ChessGame& game, NeuralNetwork* nn, player whoIsPlayed, bool highestProba = true, std::pair<float, unsigned int>* proba = nullptr, bool backpropagationTraining = true);


//TODO: could expand in the future to allow for varying topologies for regeneration of NNs.(NEAT?)
class MatchMaker
{
private:
	std::vector<std::pair<NeuralNetwork*, size_t>> m_competitors;

	size_t m_maxThreads = 16;
	size_t m_initialNNs;

	bool m_backpropagationTraining = true;

	std::vector<scoreRange> m_selectionCriteria = { scoreRange{centipawnsOrMate::makeCentipawns(0),centipawnsOrMate::makeCentipawns(200)},scoreRange{centipawnsOrMate::makeCentipawns(200),centipawnsOrMate::makeCentipawns(600)},scoreRange{centipawnsOrMate::makeCentipawns(600),centipawnsOrMate::makeMovesToMate(2)} };

	Topology m_top;
	NNInitialization m_initialRandStrat = NNInitialization();
	LearningSchedule m_learningSchedule = LearningSchedule();
	RealGameSampler m_gameSampler;

	MatchMaker();

public:

	static bool verboseOutputAndTracking;

	//generates x amount of NNs all with the same topology passed
	MatchMaker(size_t initialNNs, Topology top = QEAC_DEFAULT_TOPOLOGY);
	//pass a vector with NNs that can have any topology BUT the same input and output amount of neurons
	//also pass a default topology to generate new ones in the regeneration phases of matchmaking/simulation.
	MatchMaker(std::vector<NeuralNetwork*> initialNNs, Topology defaultTop = QEAC_DEFAULT_TOPOLOGY);


	size_t getMaxThreads()const;
	void setMaxThreads(size_t maxThreads);

	std::vector<NeuralNetwork*> getNNs();

	std::string getScoresStrings()const noexcept;

	void matchMake();

	void sortNNs();

	NeuralNetwork* getBest();

	void split();
	//creates double-1 amount of new NNs based on the ordered old ones, the one in the last position is not replicated
	//and instead a whole new random one is generated.
	void regenerate();
	
	std::stringstream *serializeMatchMaker()const;

	~MatchMaker();


};

