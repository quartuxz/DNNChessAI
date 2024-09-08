

/*
source starting here from:
https://github.com/stratzilla/connect-four/blob/master/source.cpp
HAS SOME MODIFICATIONS
*/
//
//
#include <stdio.h>
#include <iostream>
#include <vector>
#include <limits.h>
#include <array>
#include <sstream>

#include "ConnectFourTest.h"


// function declarations
void printBoard(const boardC4_t&);
int userMove(boardC4_t&);
void makeMove(boardC4_t&, int, unsigned int);
void errorMessage(int);
int aiMove(boardC4_t&, unsigned int,unsigned int);
bool winningMove(const boardC4_t&, unsigned int);
int scoreSet(std::vector<unsigned int>, unsigned int);
int tabScore(const boardC4_t&, unsigned int);
std::array<int, 2> miniMax(boardC4_t &,unsigned int, unsigned int, int, int, unsigned int);
int heurFunction(unsigned int, unsigned int, unsigned int);
void initBoard(boardC4_t&);
std::vector<boardC4_t> getPossibleMoves(boardC4_t& board, playerC4 whoIsPlayed);

// I'll be real and say this is just to avoid magic numbers
unsigned int NUM_COL = 7; // how wide is the board
unsigned int NUM_ROW = 6; // how tall
unsigned int PLAYER = 1; // playerC4/NN number
unsigned int COMPUTER = 2; // minimax AI number






/**
 * game playing function
 * loops between players while they take turns
 */
void playGame() {

	bool gameOver = false; // flag for if game is over
	unsigned int turns = 0; // count for # turns
	unsigned int currentPlayer = PLAYER; // current playerC4
	unsigned int maxDepth = 5;


	boardC4_t board; // the game board
	initBoard(board);

	printBoard(board); // print initial board
	while (!gameOver) { // while no game over state
		if (currentPlayer == COMPUTER) { // AI move
			makeMove(board, aiMove(board,turns, maxDepth), COMPUTER);
		}
		else if (currentPlayer == PLAYER) { // playerC4 move
			makeMove(board, userMove(board), PLAYER);
		}
		else if (turns == NUM_ROW * NUM_COL) { // if std::max number of turns reached
			gameOver = true;
		}
		gameOver = winningMove(board, currentPlayer); // check if playerC4 won
		currentPlayer = (currentPlayer == 1) ? 2 : 1; // switch playerC4
		turns++; // iterate number of turns
		std::cout << std::endl;
		printBoard(board); // print board after successful move
	}
	if (turns == NUM_ROW * NUM_COL) { // if draw condition
		std::cout << "Draw!" << std::endl;
	}
	else { // otherwise, someone won
		std::cout << ((currentPlayer == PLAYER) ? "AI Wins!" : "Player Wins!") << std::endl;
	}
}

/**
 * function that makes the move for the playerC4
 * @param b - the board to make move on
 * @param c - column to drop piece into
 * @param p - the current playerC4
 */
void makeMove(boardC4_t& b, int c, unsigned int p) {
	if (c == -1) {
		for (unsigned int c = 0; c < NUM_COL; c++) { // for each possible move
			if (b[NUM_ROW - 1][c] == 0) { // but only if that column is non-full
				makeMove(b,c,p);
				return;
			}
		}
	}
	// start from bottom of board going up
	for (unsigned int r = 0; r < NUM_ROW; r++) {
		if (b[r][c] == 0) { // first available spot
			b[r][c] = p; // set piece
			break;
		}
	}
}

/**
 * prompts the user for their move
 * and ensures valid user input
 * @return the user chosen column
 */
int userMove(boardC4_t &board) {
	int move = -1; // temporary
	while (true) { // repeat until proper input given
		std::cout << "Enter a column: ";
		std::cin >> move; // init move as input
		if (!std::cin) { // if non-integer
			std::cin.clear();
			std::cin.ignore(INT_MAX, '\n');
			errorMessage(1); // let user know
		}
		else if (!((unsigned int)move < NUM_COL && move >= 0)) { // if out of bounds
			errorMessage(2); // let user know
		}
		else if (board[NUM_ROW - 1][move] != 0) { // if full column
			errorMessage(3); // let user know
		}
		else { // if it gets here, input valid
			break;
		}
		std::cout << std::endl << std::endl;
	}
	return move;
}

/**
 * AI "think" algorithm
 * uses minimax to find ideal move
 * @return - the column number for best move
 */
int aiMove(boardC4_t &board, unsigned int turns, unsigned int maxDepth) {
	return miniMax(board,turns, maxDepth, 0 - INT_MAX, INT_MAX, COMPUTER)[1];
}

/**
 * Minimax implementation using alpha-beta pruning
 * @param b - the board to perform MM on
 * @param d - the current depth
 * @param alf - alpha
 * @param bet - beta
 * @param p - current playerC4
 */
std::array<int, 2> miniMax(boardC4_t& b,unsigned int turns, unsigned int d, int alf, int bet, unsigned int p) {
	/**
	 * if we've reached minimal depth allowed by the program
	 * we need to stop, so force it to return current values
	 * since a move will never (theoretically) get this deep,
	 * the column doesn't matter (-1) but we're more interested
	 * in the score
	 *
	 * as well, we need to take into consideration how many moves
	 * ie when the board is full
	 */
	if (d == 0 || d >= (NUM_COL * NUM_ROW) - turns) {
		// get current score to return
		return std::array<int, 2>{tabScore(b, COMPUTER), -1};
	}
	if (p == COMPUTER) { // if AI playerC4
		std::array<int, 2> moveSoFar = { INT_MIN, -1 }; // since maximizing, we want lowest possible value
		if (winningMove(b, PLAYER)) { // if playerC4 about to win
			return moveSoFar; // force it to say it's worst possible score, so it knows to avoid move
		} // otherwise, business as usual
		for (unsigned int c = 0; c < NUM_COL; c++) { // for each possible move
			if (b[NUM_ROW - 1][c] == 0) { // but only if that column is non-full
				boardC4_t newBoard = b;
				makeMove(newBoard, c, p); // try the move
				int score = miniMax(newBoard,turns, d - 1, alf, bet, PLAYER)[0]; // find move based on that new board state
				if (score > moveSoFar[0]) { // if better score, replace it, and consider that best move (for now)
					moveSoFar = { score, (int)c };
				}
				alf = std::max(alf, moveSoFar[0]);
				if (alf >= bet) { break; } // for pruning
			}
		}
		return moveSoFar; // return best possible move
	}
	else {
		std::array<int, 2> moveSoFar = { INT_MAX, -1 }; // since PLAYER is minimized, we want moves that diminish this score
		if (winningMove(b, COMPUTER)) {
			return moveSoFar; // if about to win, report that move as best
		}
		for (unsigned int c = 0; c < NUM_COL; c++) {
			if (b[NUM_ROW - 1][c] == 0) {
				boardC4_t newBoard = b;
				makeMove(newBoard, c, p);
				int score = miniMax(newBoard,turns, d - 1, alf, bet, COMPUTER)[0];
				if (score < moveSoFar[0]) {
					moveSoFar = { score, (int)c };
				}
				bet = std::min(bet, moveSoFar[0]);
				if (alf >= bet) { break; }
			}
		}
		return moveSoFar;
	}
}

/**
 * function to tabulate current board "value"
 * @param b - the board to evaluate
 * @param p - the playerC4 to check score of
 * @return - the board score
 */
int tabScore(const boardC4_t &b, unsigned int p) {
	int score = 0;
	std::vector<unsigned int> rs(NUM_COL);
	std::vector<unsigned int> cs(NUM_ROW);
	std::vector<unsigned int> set(4);
	/**
	 * horizontal checks, we're looking for sequences of 4
	 * containing any combination of AI, PLAYER, and empty pieces
	 */
	for (unsigned int r = 0; r < NUM_ROW; r++) {
		for (unsigned int c = 0; c < NUM_COL; c++) {
			rs[c] = b[r][c]; // this is a distinct row alone
		}
		for (unsigned int c = 0; c < NUM_COL - 3; c++) {
			for (int i = 0; i < 4; i++) {
				set[i] = rs[c + i]; // for each possible "set" of 4 spots from that row
			}
			score += scoreSet(set, p); // find score
		}
	}
	// vertical
	for (unsigned int c = 0; c < NUM_COL; c++) {
		for (unsigned int r = 0; r < NUM_ROW; r++) {
			cs[r] = b[r][c];
		}
		for (unsigned int r = 0; r < NUM_ROW - 3; r++) {
			for (int i = 0; i < 4; i++) {
				set[i] = cs[r + i];
			}
			score += scoreSet(set, p);
		}
	}
	// diagonals
	for (unsigned int r = 0; r < NUM_ROW - 3; r++) {
		for (unsigned int c = 0; c < NUM_COL; c++) {
			rs[c] = b[r][c];
		}
		for (unsigned int c = 0; c < NUM_COL - 3; c++) {
			for (int i = 0; i < 4; i++) {
				set[i] = b[r + i][c + i];
			}
			score += scoreSet(set, p);
		}
	}
	for (unsigned int r = 0; r < NUM_ROW - 3; r++) {
		for (unsigned int c = 0; c < NUM_COL; c++) {
			rs[c] = b[r][c];
		}
		for (unsigned int c = 0; c < NUM_COL - 3; c++) {
			for (int i = 0; i < 4; i++) {
				set[i] = b[r + 3 - i][c + i];
			}
			score += scoreSet(set, p);
		}
	}
	return score;
}

/**
 * function to find the score of a set of 4 spots
 * @param v - the row/column to check
 * @param p - the playerC4 to check against
 * @return - the score of the row/column
 */
int scoreSet(std::vector<unsigned int> v, unsigned int p) {
	unsigned int good = 0; // points in favor of p
	unsigned int bad = 0; // points against p
	unsigned int empty = 0; // neutral spots
	for (unsigned int i = 0; i < v.size(); i++) { // just enumerate how many of each
		good += (v[i] == p) ? 1 : 0;
		bad += (v[i] == PLAYER || v[i] == COMPUTER) ? 1 : 0;
		empty += (v[i] == 0) ? 1 : 0;
	}
	// bad was calculated as (bad + good), so remove good
	bad -= good;
	return heurFunction(good, bad, empty);
}

/**
 * my """heuristic function""" is pretty bad, but it seems to work
 * it scores 2s in a row and 3s in a row
 * @param g - good points
 * @param b - bad points
 * @param z - empty spots
 * @return - the score as tabulated
 */
int heurFunction(unsigned int g, unsigned int b, unsigned int z) {
	int score = 0;
	if (g == 4) { score += 500001; } // preference to go for winning move vs. block
	else if (g == 3 && z == 1) { score += 5000; }
	else if (g == 2 && z == 2) { score += 500; }
	else if (b == 2 && z == 2) { score -= 501; } // preference to block
	else if (b == 3 && z == 1) { score -= 5001; } // preference to block
	else if (b == 4) { score -= 500000; }
	return score;
}

/**
 * function to determine if a winning move is made
 * @param b - the board to check
 * @param p - the player to check against
 * @return - whether that player won
 */
bool winningMove(const boardC4_t& b, unsigned int p) {
	unsigned int winSequence = 0; // to count adjacent pieces
	// for horizontal checks
	for (unsigned int c = 0; c < NUM_COL - 3; c++) { // for each column
		for (unsigned int r = 0; r < NUM_ROW; r++) { // each row
			for (int i = 0; i < 4; i++) { // recall you need 4 to win
				if ((unsigned int)b[r][c + i] == p) { // if not all pieces match
					winSequence++; // add sequence count
				}
				if (winSequence == 4) { return true; } // if 4 in row
			}
			winSequence = 0; // reset counter
		}
	}
	// vertical checks
	for (unsigned int c = 0; c < NUM_COL; c++) {
		for (unsigned int r = 0; r < NUM_ROW - 3; r++) {
			for (int i = 0; i < 4; i++) {
				if ((unsigned int)b[r + i][c] == p) {
					winSequence++;
				}
				if (winSequence == 4) { return true; }
			}
			winSequence = 0;
		}
	}
	// the below two are diagonal checks
	for (unsigned int c = 0; c < NUM_COL - 3; c++) {
		for (unsigned int r = 3; r < NUM_ROW; r++) {
			for (int i = 0; i < 4; i++) {
				if ((unsigned int)b[r - i][c + i] == p) {
					winSequence++;
				}
				if (winSequence == 4) { return true; }
			}
			winSequence = 0;
		}
	}
	for (unsigned int c = 0; c < NUM_COL - 3; c++) {
		for (unsigned int r = 0; r < NUM_ROW - 3; r++) {
			for (int i = 0; i < 4; i++) {
				if ((unsigned int)b[r + i][c + i] == p) {
					winSequence++;
				}
				if (winSequence == 4) { return true; }
			}
			winSequence = 0;
		}
	}
	return false; // otherwise no winning move
}

/**
 * sets up the board to be filled with empty spaces
 * also used to reset the board to this state
 */
void initBoard(boardC4_t &board) {
	board = boardC4_t(NUM_ROW, std::vector<int>(NUM_COL));
	for (unsigned int r = 0; r < NUM_ROW; r++) {
		for (unsigned int c = 0; c < NUM_COL; c++) {
			board[r][c] = 0; // make sure board is empty initially
		}
	}
}

/**
 * prints the board to console out
 * @param b - the board to print
 */
void printBoard(const boardC4_t& b) {
	for (unsigned int i = 0; i < NUM_COL; i++) {
		std::cout << " " << i;
	}
	std::cout << std::endl << "---------------" << std::endl;
	for (unsigned int r = 0; r < NUM_ROW; r++) {
		for (unsigned int c = 0; c < NUM_COL; c++) {
			std::cout << "|";
			switch (b[NUM_ROW - r - 1][c]) {
			case 0: std::cout << " "; break; // no piece
			case 1: std::cout << "O"; break; // one playerC4's piece
			case 2: std::cout << "X"; break; // other playerC4's piece
			}
			if (c + 1 == NUM_COL) { std::cout << "|"; }
		}
		std::cout << std::endl;
	}
	std::cout << "---------------" << std::endl;
	std::cout << std::endl;
}

/**
 * handler for displaying error messages
 * @param t - the type of error to display
 */
void errorMessage(int t) {
	if (t == 1) { // non-int input
		std::cout << "Use a value 0.." << NUM_COL - 1 << std::endl;
	}
	else if (t == 2) { // out of bounds
		std::cout << "That is not a valid column." << std::endl;
	}
	else if (t == 3) { // full column
		std::cout << "That column is full." << std::endl;
	}
	std::cout << std::endl;
}

/**
 * main driver
 */
int smain(int argc, char** argv) {

	playGame(); // begin the game
	return 0; 
}
//
//
/*
source ending here from:
https://github.com/stratzilla/connect-four/blob/master/source.cpp
*/






enum gameConditionC4 {
	playing,
	firstWon,
	secondWon,
	draw
};

std::string getGameConditionString2(gameConditionC4 cond) {
	switch (cond)
	{
	case playing:
		return "playing";
		break;
	case firstWon:
		return "first victory!";
		break;
	case secondWon:
		return "second victory!";
		break;
	case draw:
		return "draw!";
		break;
	default:
		break;
	}
}



struct MatchC4 {
	size_t second;
	size_t first;
	boardC4_t gameState;
};


playerC4 flipPlayerC4(playerC4 player) {
	return player == playerC4::first ? playerC4::second : playerC4::first;
}

std::vector<float> getNumericRepresentationOfBoard(boardC4_t& board, playerC4 p_whoIsPlayed) {
	unsigned int whoIsPlayed = (unsigned int)p_whoIsPlayed;
	std::vector<float> rep;

	for (auto row:board) {
		for (auto col : row) {
			rep.push_back(col == whoIsPlayed ? 1:0);
			rep.push_back((col != whoIsPlayed && col != 0)?1:0);
			rep.push_back(col==0?1:0);
		}
	}
	return rep;
}

playerC4 getWhoToPlay(boardC4_t& board) {
	size_t count = 0;
	for (auto row : board) {
		for (auto col : row) {
			if (col != 0){
				count++;
			}
		}
	}
	if (count%2 == 0) {
		return playerC4::first;
	}
	else {
		return playerC4::second;
	}
}

std::vector<boardC4_t> getPossibleMoves(boardC4_t& board, playerC4 whoIsPlayed) {
	std::vector<boardC4_t> boards;
	for (size_t col = 0; col < board[NUM_ROW - 1].size(); col++) {
		if (board[NUM_ROW-1][col] == 0) {
			boardC4_t newBoard = board;
			makeMove(newBoard, col, (unsigned int)whoIsPlayed);
			boards.push_back(newBoard);
		}
	}
	return boards;
}

//does not guarantee atleast any amount of moves provided, but gives it a random try.
boardC4_t sampleBoard(size_t moves) {

	boardC4_t retval;

	initBoard(retval);
	for (size_t i = 0; i < moves; i++) {
		playerC4 currentPlayer = (playerC4)((i % 2) + 1);
		auto possible = getPossibleMoves(retval,currentPlayer);
		
		while (true) {
			if (possible.empty()) {
				goto skipLoops;
			}
			std::uniform_int_distribution<unsigned int> unif(0, possible.size()-1);
			float random_val = unif(NNInitialization::engine);


			if (winningMove(possible[random_val], unsigned int(currentPlayer))) {

				possible.erase(possible.begin()+random_val);


			}
			else {
				retval = possible[random_val];
				break;
			}
		}
	}
skipLoops:
	return retval;
}

#include <queue>
#include "DefsAndUtils.h"

//win/lose heuristic for connect four
float minimalC4Heuristic(const boardC4_t &brd, playerC4 plr) {
	//testing custom heuristic
	//return float(tabScore(brd, unsigned int(plr)))/100;
	return winningMove(brd, unsigned int(plr)) ? 1.0f : 0.0f;
}

void ConnectFourTest::m_generateTrainingData(float difficulty, size_t amount)
{
	std::vector<std::pair<boardC4_t*, playerC4>> data;
	//data processed by the neural network.
	std::vector<std::pair<boardC4_t*, playerC4>> NNPData;

	std::uniform_real_distribution<float> dist(0.0,1.0);
	for (size_t i = 0; i < amount; i++) {
		int diff = std::max(7 * 6 - int(difficulty),0);
		auto brd = new boardC4_t(sampleBoard(diff));
		auto plr = getWhoToPlay(*brd);
		bool generateWithNN = true;
		if (dist(*getGlobalRandomEngine())  < m_epsilon) {
			auto pos =  getPossibleMoves(*brd,plr);
			std::uniform_int_distribution<size_t> gameDist(0,pos.size()-1);
			*brd = pos[gameDist(*getGlobalRandomEngine())];
			generateWithNN = false;
		}
		data.push_back({ brd,plr });
		if (generateWithNN) {
			NNPData.push_back({ brd,plr});
		}
	}

	m_makeMovesWithNN(NNPData);

	std::vector<std::pair<connectFourPlay, float>> preparedRBEntries;


	for (auto& d : data) {
		preparedRBEntries.push_back({ {*d.first,d.second, minimalC4Heuristic(*d.first,d.second)}, 100.0f });
	}


	m_replayBuffer.insert(preparedRBEntries);

	for (auto d:data) {
		delete d.first;
	}

}

#include <map>
#include <functional>

void ConnectFourTest::m_minimaxNN(std::pair<boardC4_t*, playerC4> game, size_t depth, size_t turns)
{
	const float minScore = INT_MIN, maxScore = INT_MAX;
	const unsigned int AIPlayer=(unsigned int)game.second, opponentPlayer = (unsigned int)flipPlayerC4(game.second);
	std::function<std::pair<float, int>(boardC4_t&, unsigned int, unsigned int, unsigned int,float,float)> m_miniMax = [&minScore,&maxScore,&AIPlayer,&opponentPlayer,&m_miniMax,this](boardC4_t& b, unsigned int turns, unsigned int d, unsigned int p, float alf, float bet)->std::pair<float, int> {

		if (d == 0 || d >= (NUM_COL * NUM_ROW) - turns) {
			// get current score to return
			float score = m_nn->forwardPassGPU({ getNumericRepresentationOfBoard(b,playerC4(AIPlayer)) })[0][0];

			//return { tabScore(b, AIPlayer), -1 };
			return std::pair<float, int>{score, -1};
			
		}
		if (p == AIPlayer) {
			std::pair<float, int>moveSoFar = { minScore, -1 }; // since maximizing, we want lowest possible value
			if (winningMove(b, opponentPlayer)) {
				return moveSoFar; // force it to say it's worst possible score, so it knows to avoid move
			} // otherwise, business as usual
			for (unsigned int c = 0; c < NUM_COL; c++) { // for each possible move
				if (b[NUM_ROW - 1][c] == 0) { // but only if that column is non-full
					boardC4_t newBoard = b;
					makeMove(newBoard, c, p); // try the move
					float score = m_miniMax(newBoard, turns, d - 1, opponentPlayer,alf,bet).first; // find move based on that new board state
					//std::cout << score << std::endl;
					if (score > moveSoFar.first) { // if better score, replace it, and consider that best move (for now)
						moveSoFar = { score, (int)c };
					}
					alf = std::max(alf, moveSoFar.first);
					if (alf >= bet) { break; } // for pruning
				}
			}
			return moveSoFar; // return best possible move
		}
		else {
			std::pair<float, int> moveSoFar = { maxScore, -1 };
			if (winningMove(b, AIPlayer)) {
				return moveSoFar; // if about to win, report that move as best
			}
			for (unsigned int c = 0; c < NUM_COL; c++) {
				if (b[NUM_ROW - 1][c] == 0) {
					boardC4_t newBoard = b;
					makeMove(newBoard, c, p);
					float score = m_miniMax(newBoard, turns, d - 1, AIPlayer,alf,bet).first;
					if (score < moveSoFar.first) {
						moveSoFar = { score, (int)c };
					}
					bet = std::min(bet, moveSoFar.first);
					if (alf >= bet) { break; }
				}
			}
			return moveSoFar;
		}
	};
	auto res = m_miniMax(*game.first,(unsigned int)turns,(unsigned int)depth, AIPlayer,minScore, maxScore);

	makeMove(*game.first,res.second, AIPlayer);


}

void ConnectFourTest::m_makeMoveWithDepthAndBreadth(std::pair<boardC4_t*, playerC4> game, size_t depth, size_t breadth, bool highestTrueSampleFalse)
{

	if (depth %2 == 0) {
		depth++;
	}

	//first we evaluate if the game is in an end state
	//all games that are not proceed to next step
	std::vector < std::pair<boardC4_t*, playerC4> > candidates(breadth, {nullptr,playerC4::first});
	std::map< size_t, size_t > candidatesOriginalCandidate;
	std::vector< boardC4_t > originalCandidates;
	boardC4_t bestCandidate;
	size_t numOfCandidates = breadth;
	for (size_t o = 0; o < depth; o++) {
		std::vector<std::vector<connectFourPlay>> allPossiblePlays;
		std::vector<C4PlayOGCandidate> allPossiblePlaysFlattenedFiltered;
		if (o == 0) {
			m_makeMovesWithNN({game},nullptr,&allPossiblePlays);
		}
		else {

			m_makeMovesWithNN(std::vector < std::pair<boardC4_t*, playerC4> >(candidates.begin(),candidates.begin()+numOfCandidates),nullptr,&allPossiblePlays);
			if (o % 2 == 1) {
				for (auto& cand:candidates) {
					cand.second = flipPlayerC4(cand.second);
				}
				continue;
			}
		}

		for (size_t i = 0; i < allPossiblePlays.size(); i++) {
			size_t cnt = 0;
			for (auto& possiblePlay: allPossiblePlays[i]) {
				if (o == 0) {
					bestCandidate = possiblePlay.boardAfterPlay;
					candidatesOriginalCandidate[cnt] = cnt;
					cnt++;

					originalCandidates.push_back(possiblePlay.boardAfterPlay);
				}
				if (getPossibleMoves(possiblePlay.boardAfterPlay, flipPlayerC4(possiblePlay.whoPlayed)).size() == 0) {

					continue;
				}
				//if (winningMove(possiblePlay.boardAfterPlay,(unsigned int)game.second)) {
				//	possiblePlay.reward = 1000;
				//} 
				//if (winningMove(possiblePlay.boardAfterPlay, (unsigned int)flipPlayerC4(game.second))) {
				//	possiblePlay.reward = -1000;
				//}

				allPossiblePlaysFlattenedFiltered.push_back({ possiblePlay, candidatesOriginalCandidate[o == 0 ? cnt - 1 : i] });

			}
		}

		if (allPossiblePlaysFlattenedFiltered.empty()) {
			*game.first = bestCandidate;
			break;
		}

		std::sort(allPossiblePlaysFlattenedFiltered.begin(),allPossiblePlaysFlattenedFiltered.end(), std::greater<C4PlayOGCandidate>());
		numOfCandidates = std::min(size_t(breadth), allPossiblePlaysFlattenedFiltered.size());
		for (size_t i = 0; i < numOfCandidates; i++) {
			

			if (o == 0) {
			}
			else {
				if (candidates[i].first != nullptr) {
					delete candidates[i].first;
					candidates[i].first = nullptr;
				}


			}

			//if (allPossiblePlaysFlattenedFiltered[i].play.reward <= -999) {
			//	*game.first = bestCandidate;
			//	goto breakAllC4;
			//}

			if (i==0) {
				bestCandidate = originalCandidates[allPossiblePlaysFlattenedFiltered[i].originalCandidate];
				if (m_tracking) {
					std::cout << "best candidate: " << std::endl;
					printBoard(allPossiblePlaysFlattenedFiltered[i].play.boardAfterPlay);
				}

			}



			//if (allPossiblePlaysFlattenedFiltered[i].play.reward >= 999) {
			//	*game.first = bestCandidate;
			//	goto breakAllC4;
			//}

			candidates[i].first = new boardC4_t(allPossiblePlaysFlattenedFiltered[i].play.boardAfterPlay);
			//we flip because the person to play is opposite the person who just played
			candidates[i].second = flipPlayerC4(allPossiblePlaysFlattenedFiltered[i].play.whoPlayed);
			candidatesOriginalCandidate[i] = allPossiblePlaysFlattenedFiltered[i].originalCandidate;


		}


		if (o == depth-1) {
			//the first candidate has the candidatesOriginalCandidate[0] original candidate
			//the first candidate is also the one with the highest score.
			*game.first = bestCandidate;
			break;
		}

		//check if all the candidates have the same original candidate.
		bool allSameCandidate = true;
		size_t prevCOC = candidatesOriginalCandidate[0];
		for (auto COC : candidatesOriginalCandidate) {
			if (COC.second  != prevCOC) {
				allSameCandidate = false;
			}
		}

		//impossible to branch out of the original candidates, one candidate wins, the one that is all.
		if (allSameCandidate) {
			*game.first = bestCandidate;
			break;
		}
	}
breakAllC4:
	if (m_tracking) {
		printBoard(*game.first);
	}
	for (auto can : candidates) {
		if (can.first != nullptr) {
			delete can.first;
		}

	}

}


void ConnectFourTest::m_trainOnce()
{
	auto samples = m_replayBuffer.popSampleN(m_batchSize);
	auto originalSamples = samples;


	std::vector < float > futureRewards(samples.size(),0);
	std::vector<size_t> oppositeTurnPlaysI;
	std::vector<std::pair<boardC4_t*, playerC4>> oppositeTurnPlays;
	//first we evaluate if the game is in an end state
	//all games that are not proceed to next step
	for (size_t i = 0; i < samples.size();i++) {
		futureRewards[i] = samples[i].data.reward;
		//check for win(has a nonzero score) or tie(no more places to play)
		if (getPossibleMoves(samples[i].data.boardAfterPlay, flipPlayerC4(samples[i].data.whoPlayed)).size() == 0 || winningMove(samples[i].data.boardAfterPlay,unsigned int(samples[i].data.whoPlayed))) {
			
			continue;
		}

		oppositeTurnPlaysI.push_back(i);

		oppositeTurnPlays.push_back({&samples[i].data.boardAfterPlay,flipPlayerC4(samples[i].data.whoPlayed)});
	}

	//we evaluate the games for the opposite player
	m_makeMovesWithNN(oppositeTurnPlays, m_targetNN);


	std::vector<size_t> turnAfterOppositePlaysI;
	std::vector<std::pair<boardC4_t*, playerC4>> turnAfterOppositePlays;

	//then we check if the opponent's move won or tied andif not we play the final turn(original player's) next.
	for (size_t i = 0; i < oppositeTurnPlaysI.size(); i++) {
		auto OTPReward = minimalC4Heuristic(samples[oppositeTurnPlaysI[i]].data.boardAfterPlay, flipPlayerC4(samples[oppositeTurnPlaysI[i]].data.whoPlayed));
		futureRewards[oppositeTurnPlaysI[i]] += -m_discount *OTPReward;
		if (getPossibleMoves(samples[oppositeTurnPlaysI[i]].data.boardAfterPlay, samples[oppositeTurnPlaysI[i]].data.whoPlayed).size() == 0 || winningMove(samples[oppositeTurnPlaysI[i]].data.boardAfterPlay, unsigned int(flipPlayerC4(samples[oppositeTurnPlaysI[i]].data.whoPlayed)))) {
			continue;
		}
		turnAfterOppositePlaysI.push_back(oppositeTurnPlaysI[i]);

		turnAfterOppositePlays.push_back({ &samples[oppositeTurnPlaysI[i]].data.boardAfterPlay,samples[oppositeTurnPlaysI[i]].data.whoPlayed });
	}

	//we evaluate the final turn(first move was chosen with an exploration policy and exists in the replay buffer, second move played by opponent, third move played by original player and its evaluation gives us the next Q-Value)
	auto finalRewards = m_makeMovesWithNN(turnAfterOppositePlays,m_targetNN);

	for (size_t i = 0; i < turnAfterOppositePlaysI.size(); i++) {
		futureRewards[turnAfterOppositePlaysI[i]] += m_discount*m_discount*finalRewards[i];
	}
	

	//do the forward pass and backpropagation
	std::vector<std::vector<float>> preparedFPGames;

	for (auto &sample : originalSamples) {
		preparedFPGames.push_back(getNumericRepresentationOfBoard(sample.data.boardAfterPlay,sample.data.whoPlayed));
	}

	m_nn->startRecording();
	auto actualRewards=m_nn->forwardPassGPU(preparedFPGames);
	m_nn->endRecording();
	m_nn->selectAndDiscardRest(0,true);

	std::vector<std::vector<float>> preparedBPErrors;
	std::vector<float> trainingWieghts;
	std::vector<float> newPriorities;
	for (size_t i = 0; i < actualRewards.size(); i++) {
		float error ;
		newPriorities.push_back(std::abs(2 * (actualRewards[i][0] - futureRewards[i])));
		preparedBPErrors.push_back({2*(actualRewards[i][0]- futureRewards[i])});
		trainingWieghts.push_back(std::pow((m_replayBuffer.size() + originalSamples.size()) * originalSamples[i].prob, m_betaWeight));
		/*
		std::cout << samples[i].prob << " " << std::endl;
		std::cout << m_replayBuffer.size() << " " << std::endl;
		std::cout << samples[i].prob
		*/
	}

	m_nn->backpropagateGPU(preparedBPErrors,trainingWieghts);

	std::vector<std::pair<connectFourPlay, float>> preparedRBEntries;

	std::vector<size_t> seenIDs;
	for (size_t i = 0; i < newPriorities.size();i++) {
		if (std::find(seenIDs.begin(),seenIDs.end(), originalSamples[i].id)!=seenIDs.end()) {
			preparedRBEntries.push_back({ originalSamples[i].data, newPriorities[i] });
			seenIDs.push_back(originalSamples[i].id);
		}

	}


	m_replayBuffer.insert(preparedRBEntries);


}

void ConnectFourTest::setSaveFile(const std::string& name)
{
	m_savefile = name;
}

std::vector<float> ConnectFourTest::m_makeMovesWithNN(std::vector<std::pair<boardC4_t*,playerC4>> games, NeuralNetwork* nn, std::vector<std::vector<connectFourPlay> >* lines,bool highestTrueSampleFalse, bool backpropagationTraining, size_t ffBatchSize)
{
	if (games.empty()) {
		return {};
	}
	if (nn == nullptr) {
		nn = m_nn;
	}

	//final chosen rewards;
	std::vector<float> retval(games.size(),0);
	//first dimension is the original game, second dimension is the move
	std::queue<std::queue<std::pair<boardC4_t,playerC4>> > allPossibleMoves;
	std::vector<std::vector<boardC4_t>> allPossibleMovesCache;
	for (auto &g : games) {
		if (lines != nullptr) {
			lines->push_back({});
		}

		auto pos = getPossibleMoves(*g.first, g.second);
		allPossibleMoves.push({});
		allPossibleMovesCache.push_back(pos);
		for (auto &p : pos) {
			allPossibleMoves.back().push({ p , g.second});
		}
		
	}

	if (backpropagationTraining) {
		nn->startRecording();
	}
	//pair is for the reward first and the ID of the forward pass second.
	std::vector<std::vector<std::pair<float, size_t>> > allRewards;
	size_t  allGameN = 0;
	while (!allPossibleMoves.empty()) {
		size_t gameN = 0;
		std::vector<std::vector<float>> boardReps;
		std::vector<size_t> allPossibleMovesDim;
		while (gameN < ffBatchSize && !allPossibleMoves.empty()) {

			auto thisGame = allPossibleMoves.front();
			allPossibleMoves.pop();
			allPossibleMovesDim.push_back(0);
			while (!thisGame.empty()) {
				boardReps.push_back(getNumericRepresentationOfBoard(thisGame.front().first,thisGame.front().second));
				thisGame.pop();
				allPossibleMovesDim[allPossibleMovesDim.size()-1]++;
				gameN++;
				
			}
		}
		

		auto rewards = nn->forwardPassGPU(boardReps);

		gameN = 0;
		for (auto i : allPossibleMovesDim) {
			allRewards.push_back({});
			for (size_t o = 0; o < i; o++) {
				allRewards.back().push_back({rewards[gameN][0],gameN + allGameN});
				gameN++;
			}
		}
		allGameN += gameN;
	}



	if (highestTrueSampleFalse) {
		std::vector<size_t> toKeep;
		for (size_t i = 0; i < allRewards.size(); i++) {
			toKeep.push_back(allRewards[i][0].second);
			float maxReward = allRewards[i][0].first;
			unsigned int maxRewardIndex = 0;
			for (size_t o = 0; o < allRewards[i].size(); o++) {
				if (allRewards[i][o].first>maxReward) {
					maxReward = allRewards[i][o].first;
					maxRewardIndex = o;
					toKeep[toKeep.size() - 1] = allRewards[i][o].second;
				}

				if (lines != nullptr) {
					lines->at(i).push_back(connectFourPlay{allPossibleMovesCache[i][o],flipPlayerC4(getWhoToPlay(allPossibleMovesCache[i][o])),allRewards[i][o].first});
				}

			}
			*games[i].first = allPossibleMovesCache[i][maxRewardIndex];
			retval[i] = maxReward;

		}
		if (backpropagationTraining) {
			if (!toKeep.empty()) {
				nn->selectAndDiscardRest(toKeep);
			}

		}
	}
	//sample from probability distribution.
	else{

	}
	if (lines != nullptr && false) {
		//guaranteeing there is always a null line  that will be the only thing recorded when the board state is won or lost.
		for (size_t i = 0; i < lines->size();i++) {
			if (lines->at(i).empty()) {
				if (winningMove(*games[i].first, (unsigned int)games[i].second)) {
					lines->at(i).push_back({ *games[i].first , games[i].second,100 });
				}
				else if (winningMove(*games[i].first, (unsigned int)flipPlayerC4(games[i].second))) {
					lines->at(i).push_back({ *games[i].first , games[i].second,-100 });
				}
				else {
					lines->at(i).push_back({ *games[i].first , games[i].second,0 });
				}
			}


		}
	}



	if (!backpropagationTraining) {
		nn->endRecording();
	}
	return retval;

	/*
	//use softmax and probablity sampling to choose a move.
	std::vector<float> ratings;

	if (backpropagationTraining) {
		nn->startRecording();
	}
	float e_sum = 0;
	for (size_t i = 0; i < possibleMoves.size(); i++) {
		auto inputVec = getNumericRepresentationOfBoard(possibleMoves[i], whoIsPlayed);
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
		game = possibleMoves[highestIndex];
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
		game = possibleMoves[chosenMove];
	}

	if (backpropagationTraining) {
		nn->endRecording();
	}
	*/

}



#include <iostream>
#include <stack>



ConnectFourTest::ConnectFourTest(Topology top)
{
	m_nn = new NeuralNetwork(top,NNInitialization(),LearningSchedule());
	m_nn->m_learningSched.learningRate = 0.00025;
	m_nn->m_learningSched.constantLearningRate = true;
	m_nn->m_learningSched.useWarmup = false;
	m_targetNN = new NeuralNetwork(*m_nn);
}


void ConnectFourTest::train(size_t epochs, size_t afterTraining)
{
	size_t maxSubpasses = epochs * m_subpasses + afterTraining;
	size_t totalSubpasses = 0;
	size_t originalBetaWeight = m_betaWeight;
	//size_t originalEpsilon = m_epsilon;
	for (size_t i = 0; i < epochs; i++) {
		std::cout << "epoch: " << i << std::endl;
		//half the number of replay data as there are training passes of batch size, so that each new replay gets sampled once and then the whole buffer gets sampled randomly a few times
		for (size_t o = 1; o < 42; o++) {
			m_generateTrainingData(o, (((float)m_batchSize * m_subpasses) / 2)/41);
		}

		if (i%10 == 0) {
			save();
		}

		for (size_t o = 0; o < m_subpasses; o++) {
			m_trainOnce();
			if (totalSubpasses % m_targetLag == 0) {
				*m_targetNN = *m_nn;
			}
			m_betaWeight = originalBetaWeight + (totalSubpasses / maxSubpasses)*(1-originalBetaWeight);
			totalSubpasses++;
		}
		//m_epsilon = originalEpsilon - float(i) / epochs;
	}
	for (size_t i = 0; i < afterTraining; i++) {
		m_trainOnce();
		if (totalSubpasses % m_targetLag == 0) {
			*m_targetNN = *m_nn;
		}
		if (i%m_subpasses==0) {
			std::cout << "new epoch" << std::endl;
		}
		m_betaWeight = originalBetaWeight + (totalSubpasses / maxSubpasses) * (1 - originalBetaWeight);
		totalSubpasses++;
	}
}

#include<sstream>

#include <functional>

typedef std::function<void(boardC4_t&, size_t)> evalFunc;



//
std::array<size_t, 3> testOnce(std::stringstream &ss, evalFunc first, evalFunc second, size_t games, bool firstStarts, unsigned int toPrintLosses = 0) {
	size_t draws = 0;
	size_t wins = 0;
	size_t losses = 0;
	unsigned int printedLosses = 0;
	for (size_t i = 0; i < games; i++) {
		bool gameOver = false; // flag for if game is over
		unsigned int turns = 0; // count for # turns
		unsigned int currentPlayer = PLAYER; // current playerC4

		if (!firstStarts) {
			currentPlayer = COMPUTER;
		}
		unsigned int maxDepth = 1;


		boardC4_t board(NUM_ROW, std::vector<int>(NUM_COL)); // the game board
		initBoard(board);

		while (!gameOver) { // while no game over state
			if (turns == NUM_ROW * NUM_COL) {
				draws++;
				break;
			}
			if (currentPlayer == COMPUTER) { // AI move
				second(board,turns);
			}
			else if (currentPlayer == PLAYER) { // playerC4 move
				first(board,turns);
			}

			gameOver = winningMove(board, currentPlayer); // check if playerC4 won

			if (gameOver && currentPlayer == COMPUTER && printedLosses < toPrintLosses) {
				printBoard(board);
				printedLosses++;
			}

			currentPlayer = (currentPlayer == 1) ? 2 : 1; // switch playerC4
			turns++; // iterate number of turns
		}
		if (turns != NUM_ROW * NUM_COL-1) {
			((currentPlayer == PLAYER) ? losses++ : wins++);
		}
	}

	ss << ", won % : " << (float)wins / games << " drawn % : " << (float)draws / games << " lost % : " << (float)losses / games << std::endl;

	return {draws,wins,losses};
}

std::string ConnectFourTest::test(size_t games)
{
	using namespace std::placeholders;
	std::stringstream ss;

	//scenarios
	boardC4_t scen1;
	initBoard(scen1);
	makeMove(scen1,1,PLAYER);
	makeMove(scen1,3,COMPUTER);
	makeMove(scen1,1,PLAYER);
	makeMove(scen1,0,COMPUTER);
	makeMove(scen1,1,PLAYER);
	m_makeMovesWithNN({ {&scen1,playerC4::second} });
	makeMove(scen1, 1, PLAYER);
	ss << "scenario 1: " << (winningMove(scen1,PLAYER)?"FAILED":"PASSED") << std::endl;
	if (m_tracking) {
		std::cout << ss.str();
	}

	auto second = [](boardC4_t &board, size_t turns, size_t depth) {
		auto move = aiMove(board, turns, depth);
		makeMove(board, move, COMPUTER);
	};
	
	evalFunc first = [this](boardC4_t& board, size_t turns) {
		
		m_makeMoveWithDepthAndBreadth({ &board,playerC4::first },4,40);
		//m_makeMovesWithNN({ {&board,playerC4::first} });
		//m_minimaxNN({&board,playerC4::first },5,turns);
	};

	evalFunc randomStrat = [](boardC4_t& board, size_t turns) {
		auto possibleBoards = getPossibleMoves(board, playerC4::second);
		auto dist = std::uniform_int_distribution<size_t>(0, possibleBoards.size() - 1);
		board = possibleBoards[dist(NNInitialization::engine)];
		};

	evalFunc randomStratFirst = [](boardC4_t& board, size_t turns) {
		auto possibleBoards = getPossibleMoves(board, playerC4::first);
		auto dist = std::uniform_int_distribution<size_t>(0, possibleBoards.size() - 1);
		board = possibleBoards[dist(NNInitialization::engine)];
		};
	ss << "NN first, max depth 4";
	auto Depth4 = testOnce(ss, first, std::bind(second,_1, _2, 4), games, true,1);
	ss << "NN first, max depth 3";
	auto Depth3 = testOnce(ss, first, std::bind(second,_1, _2, 3), games, true,1);
	ss << "NN first, max depth 2";
	auto Depth2 = testOnce(ss, first, std::bind(second,_1, _2, 2), games, true,1);
	ss << "NN first, max depth 1";
	auto Depth1 = testOnce(ss,first,std::bind(second,_1,_2,1), games, true,1);

	ss << "NN first, random";
	auto vsRandom = testOnce(ss,first,randomStrat,games,true);

	ss << "Random vs Random control";
	auto randomVsRandom = testOnce(ss, randomStratFirst, randomStrat, games, true);

	return ss.str();
}

#include <fstream>



ConnectFourTest::ConnectFourTest(std::string fileName)
{
	std::ifstream NNFile;
	NNFile.open(fileName);



	std::stringstream buffer;
	buffer << NNFile.rdbuf();

	m_nn = new NeuralNetwork(buffer.str());
}


void ConnectFourTest::save()
{
	std::stringstream ss;
	ss << m_savefile << ".txt";
	std::ofstream savefile(ss.str(), std::ios::trunc);
	savefile << m_nn->serialize();

}

void ConnectFourTest::play()
{
	evalFunc AI = [this](boardC4_t& board, size_t turns) {

		//m_makeMoveWithDepthAndBreadth({ &board,playerC4::first }, 4, 40);
		m_makeMovesWithNN({ {&board,playerC4::second} });
		//m_minimaxNN({&board,playerC4::first },5,turns);
	};

	evalFunc player = [](boardC4_t& board, size_t turns) {
		printBoard(board);
		std::string moveString;
		unsigned int move;
		std::cout << "enter move: ";
		std::cin >> moveString;
		move = std::atoi(moveString.c_str());
		makeMove(board, move, PLAYER);
		};

	std::stringstream ss;
	testOnce(ss, player, AI, 1, true);


}

ConnectFourTest::~ConnectFourTest()
{
	delete m_nn;
	if (m_targetNN != nullptr) {
		delete m_targetNN;
	}

}

bool connectFourPlay::operator<(const connectFourPlay& other)
{
	return reward < other.reward;
}

bool ConnectFourTest::C4PlayOGCandidate::operator<(const C4PlayOGCandidate& other)
{
	return play < other.play;
}

bool ConnectFourTest::C4PlayOGCandidate::operator>(const C4PlayOGCandidate& other)const
{
	return play.reward > other.play.reward;
}
