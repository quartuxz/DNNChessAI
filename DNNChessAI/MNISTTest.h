#pragma once
#include<string>
#include "NeuralNetwork.cuh"


#define MNIST_DEFAULT_TOPOLOGY {28*28,500,300,200,100,10}

class MNISTTest
{
private:
	struct instance {
		uint8_t label;
		std::vector<uint8_t> data;
	};

	std::vector<instance> m_trainingInstances;
	std::vector<instance> m_testingInstances;

	NeuralNetwork m_dense;

	void m_parseCSVInstances(std::vector<instance> &instances, std::string CSVFile);

	std::vector<std::vector<float>> m_forwardPropagate(std::vector<instance*> inst);

public:
	MNISTTest(Topology top ,std::string trainFileCSV,std::string testFileCSV);
	void train(size_t epochs = 1);
	//returns the correct prediction rate
	float test();

};

