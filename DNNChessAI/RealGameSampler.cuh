#pragma once
#include <fstream>
#include <random>
#include "ChessGame.cuh"

class RealGameSampler;

struct centipawnsOrMate{
	friend class RealGameSampler;
private:
	float centipawns = -1;
	int movesToMate = -1;
	centipawnsOrMate(float cp, int moves);
public:
	centipawnsOrMate() = delete;
	static centipawnsOrMate makeMovesToMate(int moves);
	static centipawnsOrMate makeCentipawns(float centipawns);
};

typedef std::pair<centipawnsOrMate, centipawnsOrMate> scoreRange;

class RealGameSampler
{
private:
	std::ifstream m_file;
	std::string m_motherFile;
	size_t m_fileLength;
	std::default_random_engine m_engine = std::default_random_engine(static_cast<long unsigned int>(time(0)));
	std::uniform_int_distribution<size_t> m_dist;
public:
	enum selection {
		leastDepth,
		mostDepth
	};

	RealGameSampler(const std::string& motherFile);

	ChessGame sampleGame(const scoreRange & centipawnRange, selection sel);
	~RealGameSampler();
};

