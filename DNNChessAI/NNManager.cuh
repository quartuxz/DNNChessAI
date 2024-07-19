#pragma once
#include "NeuralNetwork.cuh"
#include <map>

class NNFindError : std::exception {
public:
	enum errorType {
		duplicate, notFoundName, notFoundID
	};
private:
	errorType m_et;
public:
	NNFindError(errorType et);

	errorType getErrorType()const;

	const char* what()const override;
};

class NNManager
{
private:

	std::map<std::string, NeuralNetwork*> m_NNs;
	std::vector<NeuralNetwork*> m_NNVector;
	std::vector<std::string> m_names;
public:

	NNManager();

	//provide name and pointer
	void addNN(const std::string& name, NeuralNetwork *nn);
	//provide name and serialized data
	void addNN(const std::string& name,const std::string& data);
	//with name
	NeuralNetwork* getNN(const std::string& name);
	//with ordinal for insertion
	NeuralNetwork* getNN(unsigned int ordinal);
	std::string getNNName(unsigned int ordinal);

	std::string showNNs()const;

	~NNManager();
};

