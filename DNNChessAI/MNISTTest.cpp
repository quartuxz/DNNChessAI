#include "MNISTTest.h"
#include <fstream>
#include<random>
#include <set>
#include <numeric>
#include<algorithm>
#include <iostream>


void MNISTTest::m_parseCSVInstances(std::vector<instance>& instances, std::string CSVFile)
{
	std::fstream file(CSVFile);
	instance currInstance;
	unsigned int numCount = 0;
	std::string currNum;
	while (true) {
		char next;
		file.get(next);
		if (file.eof()) {
			break;
		}


		if (next == ',') {
			if (numCount == 0) {
				currInstance.label = std::stoi(currNum);
			}
			else {
				currInstance.data.push_back(std::stoi(currNum));
			}
			currNum = "";
			numCount++;
		}
		else if(next == '\n') {
			currInstance.data.push_back(std::stoi(currNum));
			instances.push_back(currInstance);
			numCount = 0;
			currNum = "";
			currInstance = instance();
		}
		else {
			currNum.push_back(next);
		}
		
	}
}

std::vector<std::vector<float>> MNISTTest::m_forwardPropagate(std::vector<instance*> insts)
{

	std::vector<std::vector<float>> floatInRep(insts.size());

	for(size_t i = 0; i < insts.size(); i++){
		for (auto val : insts[i]->data) {
			floatInRep[i].push_back(float(val) / 255);
		}

	}
	std::vector<std::vector<float>> result = m_dense.forwardPassGPU(floatInRep,std::min(insts.size(),size_t(64)));


	for (size_t o = 0; o < result.size(); o++) {
		float e_sum = 0;
		for (size_t i = 0; i < result[o].size(); i++) {
			e_sum += std::exp(result[o][i]);
		}

		for (size_t i = 0; i < result[o].size(); i++) {
			result[o][i] = std::exp(result[o][i]) / e_sum;
		}
	}


	return result;
	
}

MNISTTest::MNISTTest(Topology top, std::string trainFileCSV, std::string testFileCSV):
	m_dense(top,NNInitialization(),LearningSchedule())
{
	m_parseCSVInstances(m_trainingInstances,trainFileCSV);
	m_parseCSVInstances(m_testingInstances,testFileCSV);
}

void MNISTTest::train(size_t epochs)
{
	//first we create the batches with random sampling
	constexpr size_t batchSize = 64;
	std::default_random_engine engine = std::default_random_engine(static_cast<long unsigned int>(time(0)));

	for (size_t o = 0; o < epochs; o++) {
		std::cout << "Epoch: "<< o << std::endl;
		std::vector<std::vector<instance*>> batches;
		std::vector<size_t> unpicked;
		for (size_t i = 0; i < m_trainingInstances.size(); i++) {
			unpicked.push_back(i);
		}
		while (!unpicked.empty()) {
			batches.push_back({});
			for (size_t i = 0; i < batchSize; i++) {
				std::uniform_int_distribution<size_t> dist(0, unpicked.size() - 1);
				if (unpicked.empty()) {
					break;
				}
				auto picked = dist(engine);
				batches.back().push_back(&m_trainingInstances[unpicked[picked]]);
				unpicked.erase(unpicked.begin() + picked);
			}
		}

		//then we run the training loop one batch at a time.
		for (auto& batch : batches) {
			std::vector<std::vector<float>> batchCostsDeriv;
			m_dense.startRecording();
			auto results = m_forwardPropagate(batch);
			for (size_t i = 0; i < results.size(); i++) {




				std::vector<float> targets(results[i].size(), 0.0);
				targets[batch[i]->label] = 1.0;

				for (size_t o = 0; o < results[i].size(); o++) {
					results[i][o] = results[i][o] - targets[o];
				}


				
			}
			m_dense.selectAndDiscardRest(0,true);
			m_dense.endRecording();
			m_dense.backpropagateGPU(results,o,100);
		}
	}


}

float MNISTTest::test()
{
	size_t correctGuesses = 0, incorrectGuesses = 0;
	for (auto &inst:m_testingInstances) {
		auto results = m_forwardPropagate({ &inst });
		unsigned int guess = std::distance(results[0].begin(), std::max_element(results[0].begin(), results[0].end()));
		//std::cout << guess << ", " << (unsigned int)inst.label<< std::endl;
		if (guess == inst.label) {
			correctGuesses++;
		}
		else {
			incorrectGuesses++;
		}
	}

	return float(correctGuesses)/(incorrectGuesses+correctGuesses);
}
