#pragma once
#include "ChessGame.cuh"
#include "NeuralNetwork.cuh"



class TextUIChess
{
public:

	enum boardDisplayType {
		mayusMinus
	};


	enum class promptMoveResullt {
		good, gameOver, terminate
	};
private:
	ChessGame m_game;
	boardDisplayType m_brdDsp;
	NeuralNetwork* m_adversary = nullptr;
	player m_playingAs = player::neither;
public:


	TextUIChess(boardDisplayType brdDsp = boardDisplayType::mayusMinus);
	
	TextUIChess(NeuralNetwork *adversary, player playerColor, boardDisplayType brdDsp = boardDisplayType::mayusMinus);


	static std::string getBoardString(board brd, boardDisplayType brdDsp);

	gameCondition getGameCondition()const;
	
	void showBoard();

	
	static std::string getShowMovesString(const std::vector<boardAndPreviousMove> &moves);

	void showMoves();

	//returns wether the user
	promptMoveResullt promptMove();


};

