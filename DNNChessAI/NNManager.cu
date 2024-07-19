#include "NNManager.cuh"
#include <sstream>

NNManager::NNManager()
{
}

void NNManager::addNN(const std::string& name, NeuralNetwork* nn)
{
	if (m_NNs.find(name) != m_NNs.end()) {
		throw NNFindError(NNFindError::duplicate);
	}
	m_NNs[name] = nn;
	m_NNVector.push_back(nn);
	m_names.push_back(name);
}

void NNManager::addNN(const std::string& name, const std::string& data)
{
	if (m_NNs.find(name) != m_NNs.end()) {
		throw NNFindError(NNFindError::duplicate);
	}
	m_NNs[name] = new NeuralNetwork(data);
	m_NNVector.push_back(m_NNs[name]);
	m_names.push_back(name);
}

NeuralNetwork* NNManager::getNN(const std::string& name)
{
	if (m_NNs.find(name) == m_NNs.end()) {
		throw NNFindError(NNFindError::notFoundName);
	}
	return m_NNs.at(name);
}

NeuralNetwork* NNManager::getNN(unsigned int ordinal)
{
	if (m_NNVector.size() <= ordinal) {
		throw NNFindError(NNFindError::notFoundID);
	}
	return m_NNVector[ordinal];
}

std::string NNManager::getNNName(unsigned int ordinal)
{
	if (m_NNVector.size() <= ordinal) {
		throw NNFindError(NNFindError::notFoundID);
	}
	return m_names[ordinal];
}

std::string NNManager::showNNs() const
{
	std::stringstream ss;
	unsigned int counter = 0;
	for (size_t i = 0; i < m_names.size(); i++)
	{
		ss << i << ".)" << m_names[i] << std::endl;
	}
	return ss.str();
}

NNManager::~NNManager()
{

	for (auto x : m_NNs) {
		delete x.second;
	}
}

NNFindError::NNFindError(errorType et):
	m_et(et)
{
}

NNFindError::errorType NNFindError::getErrorType() const
{
	return m_et;
}

const char* NNFindError::what() const
{
	switch (m_et)
	{
	case NNFindError::duplicate:
		return "NN name already found!";
		break;
	case NNFindError::notFoundName:
		return "NN name not found!";
		break;
	case NNFindError::notFoundID:
		return "NN ID not found";
		break;
	default:
		break;
	}
	return "";
}
