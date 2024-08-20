

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

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))

// function declarations
void printBoard(const boardC4_t&);
int userMove(boardC4_t&);
void makeMove(boardC4_t&, int, unsigned int);
void errorMessage(int);
int aiMove(boardC4_t&, unsigned int,unsigned int);
bool winningMove(boardC4_t&, unsigned int);
int scoreSet(std::vector<unsigned int>, unsigned int);
int tabScore(const boardC4_t&, unsigned int);
std::array<int, 2> miniMax(boardC4_t &,unsigned int, unsigned int, int, int, unsigned int);
int heurFunction(unsigned int, unsigned int, unsigned int);
void initBoard(boardC4_t&);

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
		else if (turns == NUM_ROW * NUM_COL) { // if max number of turns reached
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
				alf = max(alf, moveSoFar[0]);
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
				bet = min(bet, moveSoFar[0]);
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
 * @param p - the playerC4 to check against
 * @return - whether that playerC4 won
 */
bool winningMove(boardC4_t& b, unsigned int p) {
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



void makeMoveWithNN(boardC4_t& game, NeuralNetwork* nn, playerC4 whoIsPlayed, bool highestProba, std::pair<float, unsigned int>* proba, bool backpropagationTraining)
{

	auto possibleMoves = getPossibleMoves(game,whoIsPlayed);



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

}




gameConditionC4 matchTwoNNs(boardC4_t& game, NeuralNetwork* first, NeuralNetwork* second, std::vector < std::pair<float, unsigned int>  >* probasFirst, std::vector < std::pair<float, unsigned int>  >* probasSecond)
{

	playerC4 whoToPlay = getWhoToPlay(game);
	while (true) {

		std::pair<float, unsigned int> nextProba;
		makeMoveWithNN(game, whoToPlay == playerC4::first ? first : second, whoToPlay, true, &nextProba, true);
		if (whoToPlay == playerC4::first) {
			probasFirst->push_back(nextProba);
		}
		else {
			probasSecond->push_back(nextProba);
		}


		if (winningMove(game,(unsigned int) whoToPlay)) {
			return whoToPlay == playerC4::first ? gameConditionC4::firstWon : gameConditionC4::secondWon;
		}

		if (getPossibleMoves(game,whoToPlay).empty()) {
			return gameConditionC4::draw;
		}


		whoToPlay = whoToPlay == playerC4::first?playerC4::second:playerC4::first;
	}

	//should never reach this.
	return gameConditionC4::playing;
}


#include <iostream>
#include <stack>



void matchMakeThreadedOnce(MatchC4& match, std::vector<NeuralNetwork>& m_competitors, std::mutex& matchesLock, size_t epoch, size_t targetBatchSize = 512) {

	matchesLock.lock();
	auto &secondNN = m_competitors[match.second];
	auto &firstNN = m_competitors[match.first];


	boardC4_t& game = match.gameState;

	std::vector<std::pair<float, unsigned int> > firstProbas;
	std::vector<std::pair<float, unsigned int> > secondProbas;

	gameConditionC4 cond = matchTwoNNs(game, &firstNN, &secondNN, &firstProbas, &secondProbas);


	std::vector < std::vector<float> > costDerivFirst;

	//-1 signals tie
	float expectedFirst = -1;
	float expectedSecond = -1;

	if (cond == gameConditionC4::secondWon) {
		expectedFirst = 0;
		expectedSecond = 1;
	}
	else if (cond == gameConditionC4::firstWon) {
		expectedFirst = 1;
		expectedSecond = 0;
	}

	for (size_t i = 0; i < firstProbas.size(); i++) {
		if (expectedFirst == -1) {
			expectedFirst = 1 / firstProbas[i].second;
		}
		costDerivFirst.push_back({ (firstProbas[i].first - expectedFirst) });
	}
	//firstNN.backpropagateGPU(costDerivFirst);

	
	auto acc = firstNN.accumulateInstanceForBackprop(costDerivFirst);
	if (acc >= targetBatchSize) {
		firstNN.backpropagateGPU();
	}
	




	std::vector < std::vector<float> > costDerivSecond;

	for (size_t i = 0; i < secondProbas.size(); i++) {
		if (expectedSecond == -1) {
			expectedSecond = 1 / secondProbas[i].second;
		}
		costDerivSecond.push_back({ (secondProbas[i].first - expectedSecond) });
	}

	//secondNN.backpropagateGPU(costDerivSecond);

	
	auto acc2 = secondNN.accumulateInstanceForBackprop(costDerivSecond);
	if (acc2 >= targetBatchSize) {
		secondNN.backpropagateGPU();
	}
	


	matchesLock.unlock();
}


void matchMakeThreaded(std::stack<MatchC4>& matches, std::vector<NeuralNetwork>& m_competitors, std::mutex& matchesLock, size_t epoch) {

	//std::cout << "-";
	while (true) {
		MatchC4 thisMatch;
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


		matchMakeThreadedOnce(thisMatch, m_competitors, matchesLock,epoch);
	}
}


//multithreaded matchmaking currently does not work.
void matchMake(std::vector<NeuralNetwork>& m_competitors, size_t epoch,size_t games,size_t m_maxThreads = 1){

	//first is black and second is white
	std::stack<MatchC4> matches;
	std::vector<std::thread*> workers;

	std::mutex matchesLock;

	std::vector<boardC4_t> selectedGames;
	for (size_t i = 0; i < games; i++) {
		if (i%22==0) {
			boardC4_t game;
			initBoard(game);
			selectedGames.push_back(game);
		}
		else {
			std::uniform_int_distribution<unsigned int> unif(1, NUM_ROW*NUM_COL-1);
			float random_val = unif(NNInitialization::engine);
			selectedGames.push_back(sampleBoard(random_val));
		}

	}

	//matches are made, care is taken to not match NNs against themselves.
	for (size_t i = 0; i < m_competitors.size(); i++)
	{
		for (size_t o = i; o < m_competitors.size(); o++) {

			if (i != o) {
				for (size_t x = 0; x < 2; x++) {


					auto black = x == 0 ? i : o;
					auto white = x == 0 ? o : i;

					for (auto game : selectedGames) {
						matches.push(MatchC4{ black,white,game });
					}



				}
			}

		}
	}
	if (m_maxThreads != 1) {
		for (size_t o = 0; o < m_maxThreads; o++) {
			//matchMakeThread(matches,m_competitors,matchesLock,competitorsLock);
			workers.push_back(new std::thread([&, o]() {matchMakeThreaded(matches, m_competitors, matchesLock, epoch); }));
		}

		for (size_t i = 0; i < workers.size(); i++) {
			workers[i]->join();
			delete workers[i];
		}
	}
	else {
		matchMakeThreaded(matches, m_competitors, matchesLock,epoch);
	}

	for (auto &comp : m_competitors) {
		comp.backpropagateGPU();
		comp.increaseEpoch();
	}

}





boardC4_t ConnectFourTest::m_predictWithEnsemble(boardC4_t board, playerC4 p)
{

	auto possibleMoves = getPossibleMoves(board, p);


	std::vector<float> finalRatings(possibleMoves.size(), 0);
	for (auto &nn:m_ensemble) {
		std::vector<float> ratings(possibleMoves.size(), 0);
		for (size_t i = 0; i < possibleMoves.size(); i++) {
			auto inputVec = getNumericRepresentationOfBoard(possibleMoves[i], p);
			ratings[i] = nn.forwardPassGPU({ inputVec })[0][0];
		}
		float e_sum = 0;
		for (size_t i = 0; i < possibleMoves.size(); i++) {
			e_sum += std::exp(ratings[i]);
		}
		for (size_t i = 0; i < ratings.size(); i++) {
			ratings[i] = std::exp(ratings[i]) / e_sum;
			finalRatings[i] += ratings[i];
		}

	}

	float highestRating = 0;
	unsigned int highestIndex = 0;
	for (size_t i = 0; i < finalRatings.size(); i++) {
		float thisProba = finalRatings[i]/m_ensemble.size();
		if (thisProba > highestRating) {
			highestIndex = i;
			highestRating = thisProba;
		}
	}


	return possibleMoves[highestIndex];
}

ConnectFourTest::ConnectFourTest(Topology top, size_t players)
{
	for (size_t i = 0; i < players; i++) {
		m_ensemble.push_back(NeuralNetwork(top,NNInitialization(),LearningSchedule()));
	}
}


void ConnectFourTest::train(size_t epochs, size_t games)
{
	for (size_t i = 0; i < epochs; i++) {
		std::cout << "epoch: " << i << std::endl;
		matchMake(m_ensemble,i,games);
		if (i%10 == 0) {
			save("C4TEST");
		}

	}
}

#include<sstream>


typedef void (*evalFunc)(boardC4_t&,size_t,ConnectFourTest&);

//
std::array<size_t, 3> testOnce(std::stringstream &ss, evalFunc first, evalFunc second, ConnectFourTest& agent, size_t games, bool firstStarts) {
	size_t draws = 0;
	size_t wins = 0;
	size_t losses = 0;

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
			if (currentPlayer == COMPUTER) { // AI move
				second(board,turns,agent);
			}
			else if (currentPlayer == PLAYER) { // playerC4 move
				first(board,turns,agent);
			}
			else if (turns == NUM_ROW * NUM_COL) { // if max number of turns reached
				gameOver = true;
			}
			gameOver = winningMove(board, currentPlayer); // check if playerC4 won
			currentPlayer = (currentPlayer == 1) ? 2 : 1; // switch playerC4
			turns++; // iterate number of turns
		}
		if (turns == NUM_ROW * NUM_COL) { // if draw condition
			draws++;
		}
		else { // otherwise, someone won
			((currentPlayer == PLAYER) ? losses++ : wins++);
		}
	}

	ss << ", won % : " << (float)wins / games << " drawn % : " << (float)draws / games << " lost % : " << (float)losses / games << std::endl;

	return {draws,wins,losses};
}

std::string ConnectFourTest::test(size_t games)
{

	std::stringstream ss;


	evalFunc second = [](boardC4_t &board, size_t turns, ConnectFourTest&) {
		makeMove(board, aiMove(board, turns, 1), COMPUTER);
	};
	
	evalFunc first = [](boardC4_t& board, size_t turns, ConnectFourTest &agent) {
		board = agent.m_predictWithEnsemble(board, playerC4::first);
	};

	evalFunc randomStrat = [](boardC4_t& board, size_t turns, ConnectFourTest& agent) {
		auto possibleBoards = getPossibleMoves(board, playerC4::second);
		auto dist = std::uniform_int_distribution<size_t>(0, possibleBoards.size() - 1);
		board = possibleBoards[dist(NNInitialization::engine)];
		};
	ss << "NN first, max depth 1";
	auto Depth2 = testOnce(ss,first,second,*this,games,true);

	ss << "NN first, random";
	auto vsRandom = testOnce(ss,first,randomStrat,*this,games,true);


	return ss.str();
}

#include <fstream>



ConnectFourTest::ConnectFourTest(std::vector<std::string> fileNames)
{
	for (auto file:fileNames) {
		std::ifstream NNFile;
		NNFile.open(file);



		std::stringstream buffer;
		buffer << NNFile.rdbuf();

		m_ensemble.push_back(NeuralNetwork(buffer.str()));
	}
}


void ConnectFourTest::save(std::string nameConv)
{
	for (size_t i = 0; i < m_ensemble.size(); i++) {
		std::stringstream ss;
		ss << nameConv << "_" << i << ".txt";
		std::ofstream savefile(ss.str(), std::ios::trunc);
		savefile << m_ensemble[i].serialize();
	}

}

void ConnectFourTest::play()
{
}
