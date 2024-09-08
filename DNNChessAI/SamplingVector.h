#pragma once
#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <stdexcept>



template<typename T>
struct SamplingEntry {
	T data;
	float prob;
	float priority;
	size_t id;
};



template<typename T>
class SamplingVector
{

private:
	std::uniform_real_distribution<float> m_unif = std::uniform_real_distribution < float > (0.0, 1.0);


	size_t m_maxThreads = 20;




	float m_updateProb(float initial = 0);


	//stored type, probability to pick(they add up to 1) and initial priority
	std::vector<SamplingEntry<T>> m_distribution;

	float m_priorityGreed = 1.0f;
	size_t m_maxSize = 10000;
	size_t m_maxID = 0;

	//with replacement
	std::pair<SamplingEntry<T>*,size_t> m_sampleOnce();
	void m_removeRandomN(size_t n);
public:
	
	SamplingVector();
	SamplingVector(float priorityGreed,size_t maxSize);


	void push(const T& val, float priority);
	void insert(const std::vector<std::pair<T,float>> &other);
	SamplingEntry<T> popSampleOnce();
	//with replacement
	std::vector<SamplingEntry<T>> popSampleN(size_t n);

	void setPriorityGreed(float greed);
	float getPriorityGreed()const;

	size_t size()const;

};



template<typename T>
inline float m_updateProbThreadedPt1(typename std::vector<SamplingEntry<T>>::const_iterator begin, size_t size, float priorityGreed)
{
	float e_sum;
	for (size_t i = 0; i < size; i++) {
		e_sum += std::exp(std::pow((begin+i)->priority, priorityGreed));
	}
	return e_sum;
}

template<typename T>
inline void m_updateProbThreadedPt2(typename std::vector<SamplingEntry<T>>::iterator begin, size_t size, float e_sum, float priorityGreed)
{
	for (size_t i = 0; i < size; i++) {
		(begin + i)->prob = std::exp(std::pow((begin+i)->priority, priorityGreed)) / e_sum;
	}
}

#include <iostream>

template<typename T>
inline float SamplingVector<T>::m_updateProb(float initial)
{

	//could reactivate this branch by dividing into smaller vectors or something weird, performance gain seems unlikely.
	//cannot use iterators, not thread safe.
	if (m_distribution.size()> 100000&&false) {
		std::vector<std::thread*> workers;

		std::vector<float> e_sums(m_maxThreads, 0);
		auto& distribution = m_distribution;
		auto& priorityGreed = m_priorityGreed;
		float e_sum = 0;
		size_t offset = std::floor((double)m_distribution.size() / m_maxThreads);
		size_t remainder = m_distribution.size()%m_maxThreads;
		for (size_t i = 0; i < m_maxThreads; i++) {
			if (i== m_maxThreads-1) {

				workers.push_back(new std::thread([&]() {e_sums[i] = m_updateProbThreadedPt1<T>(distribution.begin() + offset * i, offset + remainder,priorityGreed);}));
			}
			else {
				workers.push_back(new std::thread([&]() {e_sums[i] = m_updateProbThreadedPt1<T>(distribution.begin() + offset * i, offset,priorityGreed);}));
			}



		}

		for (auto worker:workers) {
			worker->join();
			delete worker;
		}
		workers.clear();

		for (auto es : e_sums) {
			e_sum += es;
		}

		for (size_t i = 0; i < m_maxThreads; i++) {
			if (i == m_maxThreads - 1) {
				workers.push_back(new std::thread([&]() {m_updateProbThreadedPt2<T>(distribution.begin() + offset * i, offset + remainder, e_sum, priorityGreed);}));
			}
			else {
				workers.push_back(new std::thread([&]() {m_updateProbThreadedPt2<T>(distribution.begin() + offset * i, offset, e_sum,priorityGreed);}));
			}
		}
		for (auto worker : workers) {
			worker->join();
			delete worker;
		}
		workers.clear();

		return e_sum;
	}
	else {
		float e_sum = std::pow(initial, m_priorityGreed);
		for (auto& entry : m_distribution) {
			e_sum += std::exp(std::pow(entry.priority, m_priorityGreed));
		}

		for (auto& entry : m_distribution) {
			entry.prob = std::exp(std::pow(entry.priority, m_priorityGreed)) / e_sum;
		}

		return e_sum;
	}
	return 0;
}

template<typename T>
inline std::pair<SamplingEntry<T>*, size_t> SamplingVector<T>::m_sampleOnce()
{
	float random_val = m_unif(*getGlobalRandomEngine());

	float prob_sum = 0;
	size_t chosen = m_distribution.size() - 1;
	for (size_t i = 0; i < m_distribution.size(); i++) {
		prob_sum += m_distribution[i].prob;
		if (prob_sum >= random_val) {

			chosen = i;
			break;
		}
	}

	return { &m_distribution[chosen], chosen };
}

template<typename T>
inline void SamplingVector<T>::m_removeRandomN(size_t n)
{
	for (size_t i = 0; i < n; i++) {
		std::uniform_int_distribution<size_t> dist(0, m_distribution.size() - 1);
		m_distribution[dist(*getGlobalRandomEngine())] = m_distribution.back();
		m_distribution.pop_back();
	}
}

template<typename T>
inline SamplingVector<T>::SamplingVector()
{
}

template<typename T>
inline SamplingVector<T>::SamplingVector(float priorityGreed, size_t maxSize):
	m_priorityGreed(priorityGreed),
	m_maxSize(maxSize)
{
}

template<typename T>
inline void SamplingVector<T>::push(const T& val, float priority)
{
	if (m_distribution.size()+1 >= m_maxSize) {
		m_distribution.erase(m_distribution.begin());
	}
	m_distribution.push_back(SamplingEntry<T>{val,std::exp(priority) / m_updateProb(std::exp(priority)) , priority, m_maxID++});
}

template<typename T>
inline void SamplingVector<T>::insert(const std::vector<std::pair<T, float>>& other)
{
	for (auto &ele : other) {
		m_distribution.push_back(SamplingEntry<T>{ele.first,0,ele.second,m_maxID++});
	}
	m_updateProb();
}

#include "DefsAndUtils.h"

template<typename T>
inline  SamplingEntry<T> SamplingVector<T>::popSampleOnce()
{ 
	auto sam = m_sampleOnce();
	auto retval = *sam.first;
	m_removeRandom(1);
	m_updateProb();
	return retval;
}

template<typename T>
inline std::vector<SamplingEntry<T>> SamplingVector<T>::popSampleN(size_t n)
{
	if (n > m_distribution.size()) {
		throw std::invalid_argument("sampling more elements than there are in the distribution!");
	}

	if (m_distribution.size() + n > m_maxSize) {
		//m_distribution.erase(m_distribution.begin(),m_distribution.begin()+((m_distribution.size() + n)-m_maxSize));
		m_removeRandomN((m_distribution.size() + n) - m_maxSize);
	}


	std::vector<SamplingEntry<T>>  retval;
	std::vector<size_t> toRemove;

	for (size_t i = 0; i < n; i++) {
		auto sam = m_sampleOnce();
		retval.push_back(*sam.first);
		toRemove.push_back(sam.second);
	}

	std::vector<size_t> alreadyRemoved;

	for (auto idx:toRemove) {
		if (std::find(alreadyRemoved.begin(), alreadyRemoved.end(), idx) != alreadyRemoved.end()) {
			m_distribution[idx] = m_distribution.back();
			m_distribution.pop_back();
			alreadyRemoved.push_back(idx);
		}


	}

	m_updateProb();

	return retval;
}

template<typename T>
inline void SamplingVector<T>::setPriorityGreed(float greed)
{
	m_priorityGreed = greed;
	m_updateProb();
}

template<typename T>
inline float SamplingVector<T>::getPriorityGreed() const
{
	return m_priorityGreed;
}

template<typename T>
inline size_t SamplingVector<T>::size() const
{
	return m_distribution.size();
}
