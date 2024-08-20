#include "MatchMaker.cuh"

#include "TextUIChess.cuh"
#include "NeuralNetwork.cuh"

#include <map>
#include <iostream>

bool MatchMaker::verboseOutputAndTracking = true;


gameCondition matchTwoNNs(ChessGame& game, NeuralNetwork* black, NeuralNetwork* white, std::vector < std::pair<float, unsigned int>  > *probasWhite, std::vector < std::pair<float, unsigned int>  >* probasBlack)
{
	
	gameCondition retval = gameCondition::playing;
	player whoToPlay = game.getWhoToPlay();
	while (true) {

		//std::cout << TextUIChess::getBoardString(game->getCurrentBoard(),TextUIChess::boardDisplayType::mayusMinus) << std::endl;
		if (probasWhite != nullptr && probasBlack != nullptr) {
			std::pair<float, unsigned int> nextProba;
			retval = makeMoveWithNN(game, whoToPlay == player::white ? white : black, whoToPlay,true,  &nextProba, true);
			if (retval != gameCondition::playing) {
				break;
			}
			if (whoToPlay == player::white) {
				probasWhite->push_back(nextProba);
			}
			else {
				probasBlack->push_back(nextProba);
			}

		}
		else {
			retval = makeMoveWithNN(game, whoToPlay == player::white ? white : black, whoToPlay);
			if (retval != gameCondition::playing) {
				break;
			}
		}



		whoToPlay = flipColor(whoToPlay);
	}
	
	return retval;
}

gameCondition makeMoveWithNN(ChessGame &game, NeuralNetwork* nn, player whoIsPlayed, bool highestProba,std::pair<float, unsigned int>  *proba, bool backpropagationTraining)
{

	gameCondition retval;
	auto possibleMoves = game.getPossibleBoards(&retval);
	if (retval != gameCondition::playing) {
		return retval;
	}



	//use softmax and probablity sampling to choose a move.
	std::vector<float> ratings;
	
	if (backpropagationTraining) {
		nn->startRecording();
	}
	float e_sum = 0;
	for (size_t i = 0; i < possibleMoves.size(); i++) {
		auto inputVec = getNumericRepresentationOfBoard(possibleMoves[i].first, whoIsPlayed);
		game.addExtraInputs(inputVec,whoIsPlayed);
		//the most computationally intensive operation.
		ratings.push_back(nn->forwardPassGPU({ inputVec })[0][0]);
		 e_sum += std::exp(ratings.back());
	}

	
	if (highestProba) {
		float highestRating = 0;
		unsigned int highestIndex = 0;
		for (size_t i = 0; i < ratings.size(); i++) {
			float thisProba = std::exp(ratings[i]) / e_sum;
			if (thisProba > highestRating) {
				highestIndex = i;
				highestRating = thisProba;
			}
		}
		game.setNext(possibleMoves[highestIndex]);
		if (backpropagationTraining) {
			nn->selectAndDiscardRest(highestIndex);
		}
		if (proba != nullptr) {
			*proba = std::make_pair(highestRating, ratings.size());
		}

	}
	else {
		std::uniform_real_distribution<float> unif(0, 1);
		float random_val = unif(NNInitialization::engine);

		size_t chosenMove = 0;
		float prob_sum = 0;
		for (size_t i = 0; i < ratings.size(); i++) {
			float thisProba = std::exp(ratings[i]) / e_sum;
			prob_sum += thisProba;
			if (prob_sum >= random_val) {
				if (proba != nullptr) {
					*proba = std::make_pair(thisProba, ratings.size());
				}

				chosenMove = i;
				break;
			}
		}
		if (backpropagationTraining) {
			nn->selectAndDiscardRest(chosenMove);
		}
		game.setNext(possibleMoves[chosenMove]);
	}

	if (backpropagationTraining) {
		nn->endRecording();
	}


	return gameCondition::playing;
}

MatchMaker::MatchMaker():
	m_gameSampler(R"(C:\Users\Administrator\Desktop\c++\DNNChessAI\DNNChessAI\lichess_db_eval.jsonl)")
{
}

MatchMaker::MatchMaker(size_t initialNNs, Topology top):
	MatchMaker()
{
	m_initialNNs = initialNNs;
	m_top = top;
	for (size_t i = 0; i < initialNNs; i++) {
		m_competitors.push_back(std::make_pair(new NeuralNetwork(top, m_initialRandStrat,LearningSchedule()),0));
	}

}

MatchMaker::MatchMaker(std::vector<NeuralNetwork*> initialNNs, Topology top):
	MatchMaker()
{
	m_initialNNs = initialNNs.size();
	m_top = top;
	for (size_t i = 0; i < initialNNs.size(); i++)
	{
		m_competitors.push_back(std::make_pair(initialNNs[i],0));
	}
}

void MatchMaker::setGamesPerEpoch(size_t games)
{
	m_targetGamesPerEpoch = games;
}

size_t MatchMaker::getMaxThreads() const
{
	return m_maxThreads;
}

void MatchMaker::setMaxThreads(size_t maxThreads)
{
	m_maxThreads = maxThreads;
}

#include <iostream>
#include <stack>



void addScores(std::vector<std::pair<NeuralNetwork*, size_t>>& m_competitors, size_t blackIndex, size_t whiteIndex, gameCondition cond) {
	if (cond == gameCondition::blackVictory) {
		m_competitors[blackIndex].second += 2;
	}
	else if (cond == gameCondition::whiteVictory) {
		m_competitors[whiteIndex].second += 2;
	}
	//tie
	else {
		m_competitors[blackIndex].second += 1;
		m_competitors[whiteIndex].second += 1;
	}
}

void matchMakeThreadedOnce(Match &match,std::vector<std::pair<NeuralNetwork*, size_t>>& m_competitors, std::mutex &matchesLock, bool backpropagationTraining = true, size_t epoch = 0, size_t targetBatchSize = 1024 ) {
	
	matchesLock.lock();
	auto blackNN = m_competitors[match.black].first;
	auto whiteNN = m_competitors[match.white].first;
	matchesLock.unlock();

	ChessGame &game = match.gameState;

	std::vector<std::pair<float, unsigned int> > whiteProbas;
	std::vector<std::pair<float, unsigned int> > blackProbas;
	if (backpropagationTraining) {
		matchesLock.lock();
	}
	gameCondition cond = matchTwoNNs(game, blackNN,whiteNN,&whiteProbas,&blackProbas);
	
	if (backpropagationTraining) {
		std::vector < std::vector<float> > costDerivWhite;

		//-1 signals tie
		float expectedWhite = -1;
		float expectedBlack = -1;

		if (cond == gameCondition::blackVictory) {
			expectedWhite = 0;
			expectedBlack = 1;
		}
		else if (cond == gameCondition::whiteVictory) {
			expectedWhite = 1;
			expectedBlack = 0;
		}

		for (size_t i = 0; i < whiteProbas.size(); i++) {
			if (expectedWhite == -1) {
				expectedWhite = 1 / whiteProbas[i].second;
			}
			costDerivWhite.push_back({ (whiteProbas[i].first - expectedWhite) });
		}

		if (!costDerivWhite.empty()) {
			auto acc = whiteNN->accumulateInstanceForBackprop(costDerivWhite);
			if (acc >= targetBatchSize) {
				whiteNN->backpropagateGPU();
			}

		}



		std::vector < std::vector<float> > costDerivBlack;

		for (size_t i = 0; i < blackProbas.size(); i++) {
			if (expectedBlack == -1) {
				expectedBlack = 1 / blackProbas[i].second;
			}
			costDerivBlack.push_back({ (blackProbas[i].first - expectedBlack) });
		}
		if (!costDerivBlack.empty()) {
			auto acc = blackNN->accumulateInstanceForBackprop(costDerivBlack);
			if (acc >= targetBatchSize) {
				blackNN->backpropagateGPU();
			}
		}

		matchesLock.unlock();
	}


	matchesLock.lock();

	if (MatchMaker::verboseOutputAndTracking) {
		std::cout << " " << match.white << " vs " << match.black << " (first white second black): " << getGameConditionString(cond) << " " << std::endl;
	}

	addScores(m_competitors, match.black,match.white,cond);
	matchesLock.unlock();
}


void matchMakeThreaded(std::stack<Match>& matches, std::vector<std::pair<NeuralNetwork*, size_t>> &m_competitors, std::mutex &matchesLock,bool backpropagationTraining = true, size_t epoch = 0, size_t targetBatchSize = 1024) {
	
	//std::cout << "-";
	while (true) {
		Match thisMatch;
		matchesLock.lock();
		if (matches.empty()) {
			
			matchesLock.unlock();
			break;
		}
		else {
			thisMatch = matches.top();

			matches.pop();
		}
		matchesLock.unlock();





		matchMakeThreadedOnce(thisMatch, m_competitors,matchesLock,backpropagationTraining,epoch,targetBatchSize);
	}
}


#include <algorithm>
#include <chrono>

#define START_CHRONO auto start = std::chrono::high_resolution_clock::now();
#define END_CHRONO_LOG auto finish = std::chrono::high_resolution_clock::now();\
						std::cout << std::endl;\
						std::cout << "time taken in milliseconds: " <<std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << std::endl;

std::string MatchMaker::getScoresStrings() const noexcept
{
	std::stringstream ss;
	ss << "scores: " << std::endl;
	for (auto x : m_competitors)
	{
		ss << x.second << std::endl;
	}
	return ss.str();
}

std::vector<NeuralNetwork*> MatchMaker::getNNs()
{
	std::vector<NeuralNetwork*> retval;
	for (auto x : m_competitors)
	{
		retval.push_back(x.first);
	}
	return retval;
}

void MatchMaker::matchMake(bool backpropagationTraining)
{
	if (MatchMaker::verboseOutputAndTracking) {
		std::cout << "start of matchmake" << std::endl;
	}
	using milli = std::chrono::milliseconds;


	//first is black and second is white
	std::stack<Match> matches;
	std::vector<std::thread*> workers;

	std::mutex matchesLock;

	std::vector<ChessGame> selectedGames;
	for (size_t i = 0; i < m_targetGamesPerEpoch/m_selectionCriteria.size(); i++) {
		for (auto& criteria : m_selectionCriteria) {
			//selectedGames.push_back(ChessGame());
			selectedGames.push_back(m_gameSampler.sampleGame(criteria, RealGameSampler::leastDepth));
		}
	}


	//matches are made, care is taken to not match NNs against themselves.
	for (size_t i = 0; i < m_initialNNs; i++)
	{
		for (size_t o = i; o < m_initialNNs; o++) {

			if (i != o) {
				for (size_t x = 0; x < 2; x++) {


					auto black = x == 0 ? i : o;
					auto white = x == 0 ? o : i;

					for (auto game : selectedGames) {
						matches.push(Match{ black,white,game });
					}



				}
			}

		}
	}

	START_CHRONO
	if (m_maxThreads != 1) {
		for (size_t o = 0; o < m_maxThreads; o++) {
			//matchMakeThread(matches,m_competitors,matchesLock,competitorsLock);
			workers.push_back(new std::thread([&, o]() {matchMakeThreaded(matches, m_competitors, matchesLock,backpropagationTraining,m_currentEpoch, m_targetBatchSize); }));
		}

		for (size_t i = 0; i < workers.size(); i++) {
			workers[i]->join();
			delete workers[i];
		}
	}
	else {
		matchMakeThreaded(matches, m_competitors, matchesLock,backpropagationTraining,m_currentEpoch,m_targetBatchSize);
	}

	//we do this to ensure instances that were computed but exceeded the the last batch max are backpropagated.
	for (auto comp : m_competitors) {
		
		comp.first->backpropagateGPU();
		comp.first->increaseEpoch();
	}


	
	m_currentEpoch++;


	END_CHRONO_LOG



	/*
	auto start = std::chrono::high_resolution_clock::now();
	matchMakeThreaded(matches, m_competitors, matchesLock, competitorsLock);

	auto finish = std::chrono::high_resolution_clock::now();

	std::cout << std::endl;
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << std::endl;
	*/
	//return;



}

#include <algorithm>


bool sortBySec(const std::pair<NeuralNetwork*, size_t> &a, const std::pair<NeuralNetwork*, size_t>& b) {
	return (a.second > b.second);
}


void MatchMaker::sortNNs()
{
	if (MatchMaker::verboseOutputAndTracking) {
		std::cout << "start of sort" << std::endl;
	}
	std::stable_sort(m_competitors.begin(), m_competitors.end(), sortBySec);
	if (verboseOutputAndTracking) {
		std::cout << "scores: " << std::endl;
		for (auto x : m_competitors)
		{
			std::cout << x.second << std::endl;
		}
	}
}

NeuralNetwork* MatchMaker::getBest()
{
	return m_competitors[0].first;
}

void MatchMaker::split()
{
	if(MatchMaker::verboseOutputAndTracking){
		std::cout << "start of split" << std::endl;
	}
	for (size_t i = m_competitors.size()/2; i < m_competitors.size(); i++) {
		delete m_competitors[i].first;
	}

	m_competitors.erase(m_competitors.begin()+ m_competitors.size()/2, m_competitors.end());

}

void MatchMaker::regenerate()
{
	if(MatchMaker::verboseOutputAndTracking){
		std::cout << "start of regenerate" << std::endl;
	}
	//resetting all scores
	for (auto &x : m_competitors) {
		x.second = 0;
	}

	//competitors are cut in half during split()
	size_t initialCompetitorsSize = m_competitors.size();

	//reset all scores and create new competitors, mutated duplicates from the first ones
	for (size_t i = 0; i < initialCompetitorsSize-1; i++)
	{
		NeuralNetwork* newNN;


		newNN = new NeuralNetwork(*m_competitors[i].first);

		newNN->addRandomWeights();
		m_competitors.push_back(std::make_pair(newNN,0));

	}
	

	//generate a a completely random new competitor
	NeuralNetwork* newNN;

	newNN = new NeuralNetwork(m_top,m_initialRandStrat,m_learningSchedule);

	m_competitors.push_back(std::make_pair(newNN,0));
	
}

std::stringstream *MatchMaker::serializeMatchMaker() const
{
	std::stringstream *ss = new std::stringstream;
	for (auto nn : m_competitors)
	{

		(*ss) << nn.first->serialize() << std::endl << std::endl;
	}

	return ss;
}

MatchMaker::~MatchMaker()
{
	for (auto x : m_competitors) {
		delete x.first;
	}
}
