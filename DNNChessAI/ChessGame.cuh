#pragma once

#include <array>
#include <vector>
#include <string>
#include <optional>





enum class gameCondition: uint8_t {
	playing, tieByStalemate, tieBy50Moves, tieByThreeFoldRepetition, whiteVictory, blackVictory
};

std::string getGameConditionString(gameCondition condition);

enum class chessPiece : uint8_t {
	empty, blackPawn, blackRook, blackKnight, blackBishop,blackKing,blackQueen, whitePawn, whiteRook,whiteKnight,whiteBishop, whiteKing, whiteQueen
};

enum class colorlessChessPiece : uint8_t {
	pawn ,rook, knight, bishop, king, queen
};

typedef std::array<std::array<chessPiece,8>, 8> board;


//IMPORTANT: first coordinate corresponds to the row and second to the column
typedef std::pair<int8_t, int8_t> boardCoords;


enum class player:uint8_t {
	neither, white, black
};


player flipColor(player who);

struct chessMove {
	enum moveTypes:uint8_t {
		doublePawn, castle, normal, capture, promotion, captureAndPromotion, enPassant, notAMove
	}moveType;

	enum promotionTypes:uint8_t {
		notPromotion, toRook, toKnight, toBishop, toQueen
	}promotionTo = notPromotion;

	colorlessChessPiece initalPiece;

	boardCoords whereFrom;
	boardCoords whereTo;
	player who;
	chessMove(moveTypes p_moveType, boardCoords p_whereFrom, boardCoords p_whereTo, player p_who, colorlessChessPiece piece);
	chessMove();
	std::string getStringRepresentation()const;
};

typedef std::pair<board, chessMove> boardAndPreviousMove;


class ChessGame
{
private:
	board m_current;

	std::vector<board> m_pastBoards;

	//the back is the latest move
	std::vector<chessMove> m_moves = {chessMove()};
	player m_whoToPlay = player::white;

	size_t m_movesWithoutCaptureOrPawnMove = 0;
	bool m_repeatedPosition = false;


	bool m_WHasLongCastlingRights = true;
	bool m_WHasShortCastlingRights = true;
	bool m_BHasLongCastlingRights = true;
	bool m_BHasShortCastlingRights = true;

	bool m_WCanCastleLong = false;
	bool m_WCanCastleShort = false;
	bool m_BCanCastleLong = false;
	bool m_BCanCastleShort = false;

	bool m_WMovedKing = false;
	bool m_WMovedLongRook = false;
	bool m_WMovedShortRook = false;
	bool m_BMovedKing = false;
	bool m_BMovedLongRook = false;
	bool m_BMovedShortRook = false;

	std::vector<boardAndPreviousMove> m_getPossibleMovesForBoard(const boardAndPreviousMove &brd, player whoToPlay)const;

	bool m_checkWouldCaptureKing(const boardAndPreviousMove &brd)const;

	void m_setCanCastle(const boardAndPreviousMove& brd, player whoToPlay);

public:


	player getWhoToPlay()const;

	bool getRepeatedPosition()const noexcept;

	size_t getMovesWithoutCaptureOrPawnMove()const noexcept;


	//load from FEN code
	ChessGame(const std::string& fen);

	//default starting board
	ChessGame();

	//returns all possible moves for current board and player, optionally pass gameCondition pointer to know what condition is shown.
	std::vector<boardAndPreviousMove> getPossibleBoards(gameCondition *condition = nullptr)const;
	void setNext(boardAndPreviousMove brdMove);

	board getCurrentBoard()const noexcept;

	void addExtraInputs(std::vector<float>& pastInputs, player whoIsPlayed)const;
};



board flipBoard(const board &brd);

std::vector<float> getNumericRepresentationOfBoard(board brd, player whoToPlay);







