#include "RealGameSampler.cuh"
#include <algorithm>
#include <iostream>
#include "json.hpp"


// for convenience
using json = nlohmann::json;


centipawnsOrMate::centipawnsOrMate(float cp, int moves):
	centipawns(cp),
	movesToMate(moves)
{
}

centipawnsOrMate centipawnsOrMate::makeMovesToMate(int moves)
{
	return centipawnsOrMate(-1,moves);
}

centipawnsOrMate centipawnsOrMate::makeCentipawns(float centipawns)
{
	return centipawnsOrMate(centipawns,-1);
}

centipawnsOrMate::centipawnsOrMate()
{
}

RealGameSampler::RealGameSampler(const std::string& motherFile):
	m_file(motherFile),
	m_motherFile(motherFile)
{
	m_file.seekg(0, m_file.end);
	m_fileLength = m_file.tellg();
	m_file.seekg(0, m_file.beg);
	m_dist = std::uniform_int_distribution<size_t>(0, m_fileLength);
	m_file.clear(); // clear bad state after eof
	m_file.close();

}

ChessGame RealGameSampler::sampleGame(const std::pair<centipawnsOrMate, centipawnsOrMate>& centipawnRange, selection sel)
{
	m_file.open(m_motherFile);
	while (true) {
		m_file.seekg(m_dist(m_engine));
		std::string discovered;
		unsigned int num_of_brackets = 0;
		bool foundFen = false;
		while (true) {
			char next;
			m_file.get(next);
			if (next == '{' && !foundFen) {
				num_of_brackets = 1;
				discovered = "";
			}
			discovered.push_back(next);
			if (foundFen || discovered.find("fen") != std::string::npos) {
				if (num_of_brackets == 0) {
					goto restart;
				}
				foundFen = true;
				if (next == '{') {
					num_of_brackets += 1;
				}
				else if (next == '}') {
					num_of_brackets -= 1;
				}
				if (num_of_brackets == 0) {
					break;
				}
			}
			if (m_file.eof()) {
restart:
				m_file.seekg(m_dist(m_engine));
				m_file.clear();
				discovered = "";
				num_of_brackets = 0;
				foundFen = false;
				continue;
			}
		}
		json gameJSON = json::parse(discovered);

		std::vector<int> depths;
		std::vector<centipawnsOrMate> scores;
		for (auto& eval : gameJSON["evals"]) {
			int depth = 0;
			eval["depth"].get_to(depth);
			depths.push_back(depth);
			centipawnsOrMate score = centipawnsOrMate{-1,-1};
			for (auto& lineAndScore : eval["pvs"]) {

				for (auto& field : lineAndScore.items()) {
					if (field.key() == "cp") {

						if (score.movesToMate == -1) {
							int centipawn = 0;
							lineAndScore["cp"].get_to(centipawn);
							//read as absoulte value as score could be for white or black
							centipawn = abs(centipawn);
							if (centipawn > score.centipawns) {
								score.centipawns = centipawn;
							}
						}
						
					}
					else if(field.key() == "mate") {
						int movesToMate = -1;
						lineAndScore["mate"].get_to(movesToMate);
						//read as absoulte value as score could be for white or black
						movesToMate = abs(movesToMate);
						if (movesToMate > score.movesToMate) {
							score.movesToMate = movesToMate;
							score.centipawns = -1;
						}
						
					}
				}

			}

			scores.push_back(score);
		}

		auto checkScores = [&](centipawnsOrMate &score) {
			if (centipawnRange.first.centipawns >= 0) {
				if (score.centipawns >= 0) {
					if (score.centipawns < centipawnRange.first.centipawns) {
						return false;
					}
				}
				else {
					return false;
				}
			}
			else {
				if (score.movesToMate > centipawnRange.first.movesToMate) {
					return false;
				}
			}


			if (centipawnRange.second.centipawns >= 0) {
				if (score.centipawns >= 0) {
					if (score.centipawns > centipawnRange.second.centipawns) {
						return false;
					}
				}
				else {
					return false;
				}
			}
			else {
				if (score.movesToMate >= 0 && score.movesToMate < centipawnRange.second.movesToMate) {
					return false;
				}
			}
			return true;
		};

		if (sel == selection::leastDepth) {
			unsigned int depth = depths[0];
			size_t ID=0;
			for (size_t i = 0; i < depths.size(); i++) {
				if (sel == selection::leastDepth ? depths[i] < depth : depths[i] > depth) {
					ID = i;
					depth = depths[i];
				}
			}
			if (checkScores(scores[ID])) {
				std::string fen;
				gameJSON["fen"].get_to(fen);
				m_file.close();
				return ChessGame(fen);
			}
		}
	}
	m_file.close();
	return ChessGame();
}

RealGameSampler::~RealGameSampler()
{

}
