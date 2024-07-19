#include "ChessGame.cuh"
#include <sstream>
#include <map>
#include <stdexcept>
#include <optional>

#include <iostream>




player getPieceColor(chessPiece piece) {
    if ((size_t)piece == 0) {
        return player::neither;
    }
    if ((size_t)piece > 0 && (size_t)piece <= 6) {
        return player::black;
    }
    if ((size_t)piece > 6 && (size_t)piece <= 12) {
        return player::white;
    }
}

//just places the piece on the given board coordinate and board.
board makeMove(board brd, boardCoords coords, chessPiece piece) {
    brd[coords.first][coords.second] = piece;
    return brd;
}

chessPiece makePiece(colorlessChessPiece piece, player whoToPlay) {
    size_t start = 1;
    if (whoToPlay == player::white) {
        start = 7;
    }
    return (chessPiece)((size_t)piece + start);
}


//NOTE: does not take empty piece
colorlessChessPiece makeColorlessPiece(chessPiece piece) {


    switch (piece)
    {
    case chessPiece::empty:
        throw std::invalid_argument("piece is empty piece and cant be converted!");
        break;
    case chessPiece::whitePawn:
    case chessPiece::blackPawn:
        return colorlessChessPiece::pawn;
        break;
    case chessPiece::whiteRook:
    case chessPiece::blackRook:
        return colorlessChessPiece::rook;
        break;
    case chessPiece::whiteKnight:
    case chessPiece::blackKnight:
        return colorlessChessPiece::knight;
        break;
    case chessPiece::whiteBishop:
    case chessPiece::blackBishop:
        return colorlessChessPiece::bishop;
        break;
    case chessPiece::whiteKing:
    case chessPiece::blackKing:
        return colorlessChessPiece::king;
        break;
    case chessPiece::whiteQueen:
    case chessPiece::blackQueen:
        return colorlessChessPiece::queen;
        break;
    default:
        break;
    }
}


#define SIMPLE_MOVE(moves, brd, coordsTo, coordsFrom, piece, whoToPlay, moveType) moves.push_back( boardAndPreviousMove{makeMove(brd, coordsTo, makePiece(piece, whoToPlay)), chessMove(moveType,coordsFrom,coordsTo,whoToPlay,piece)})

#define PAWN_PROMOTION(moves, brd, coordsTo, coordsFrom, whoToPlay, moveType)   SIMPLE_MOVE(moves, brd, coordsTo, coordsFrom, colorlessChessPiece::rook, whoToPlay, chessMove::promotion);\
                                                                                moves.back().second.promotionTo = chessMove::toRook;\
                                                                                SIMPLE_MOVE(moves, brd, coordsTo, coordsFrom, colorlessChessPiece::queen, whoToPlay, chessMove::promotion);\
                                                                                moves.back().second.promotionTo = chessMove::toQueen;\
                                                                                SIMPLE_MOVE(moves, brd, coordsTo, coordsFrom, colorlessChessPiece::bishop, whoToPlay, chessMove::promotion);\
                                                                                moves.back().second.promotionTo = chessMove::toBishop;\
                                                                                SIMPLE_MOVE(moves, brd, coordsTo, coordsFrom, colorlessChessPiece::knight, whoToPlay, chessMove::promotion);\
                                                                                moves.back().second.promotionTo = chessMove::toKnight;\




std::map<gameCondition, std::string> gameConditionToString = {
    {gameCondition::blackVictory, "black victory"},
    {gameCondition::whiteVictory, "white victory"},
    {gameCondition::playing, "playing"},
    {gameCondition::tieBy50Moves, "tie by 50 moves"},
    {gameCondition::tieByStalemate, "tie by stalemate"},
    {gameCondition::tieByThreeFoldRepetition, "tie by three fold repetition"}
};


std::string getGameConditionString(gameCondition condition)
{
    return gameConditionToString[condition];
}

player flipColor(player who) {
    return (who == player::white ? player::black : player::white);
}


//get all pawn moves for a given position(and a piece->color therein) in a board.
std::vector<boardAndPreviousMove> getPawnMoves(board brd, const chessMove &previousMove, boardCoords coords, player whoToPlay) {
    std::vector<boardAndPreviousMove> retval;
    brd[coords.first][coords.second] = chessPiece::empty;


    int up = -1;
    chessPiece pawn = chessPiece::whitePawn;
    if (whoToPlay == player::black) {
        pawn = chessPiece::blackPawn;
        up *= -1;
    }

    auto doubleMoveCoords = boardCoords(coords.first + up * 2, coords.second);
    auto oneUp = boardCoords(coords.first + up, coords.second);

    //we take advantage of the fact a pawn can never be in the promotion rank to make the code simpler and with less checks

    //promotion
    //we check if moving forward will result moving into promotion rank
    if (oneUp.first == 0 || oneUp.first == 7) {


        if (brd[oneUp.first][oneUp.second] == chessPiece::empty) {
            PAWN_PROMOTION(retval, brd, oneUp, coords, whoToPlay, chessMove::promotion);
        }
    }
    //normal vertical-forward moves that are not promotion
    else {

        if (brd[oneUp.first][oneUp.second] == chessPiece::empty) {
            SIMPLE_MOVE(retval, brd, oneUp, coords, colorlessChessPiece::pawn, whoToPlay, chessMove::normal);
        }


        if (whoToPlay == player::white && coords.first == 6) {
            if (brd[doubleMoveCoords.first][doubleMoveCoords.second] == chessPiece::empty) {
                SIMPLE_MOVE(retval, brd, doubleMoveCoords, coords, colorlessChessPiece::pawn, whoToPlay, chessMove::doublePawn);

            }

        }
        else if (whoToPlay == player::black && coords.first == 1) {
            if (brd[doubleMoveCoords.first][doubleMoveCoords.second] == chessPiece::empty) {
                SIMPLE_MOVE(retval, brd, doubleMoveCoords, coords, colorlessChessPiece::pawn, whoToPlay, chessMove::doublePawn);
            }
        }
    }

    //return retval;

    //we do the diagonal capture moves
    //first we bound check
    if (coords.second + 1 <= 7) {
        //diagonal to the right
        auto diagRight = boardCoords(coords.first + up, coords.second + 1);
        auto rightPieceColor = getPieceColor(brd[diagRight.first][diagRight.second]);
        if (rightPieceColor != whoToPlay && rightPieceColor != player::neither) {
            //promotion moves
            if (diagRight.first == 0 || diagRight.first == 7) {
                PAWN_PROMOTION(retval, brd, diagRight, coords, whoToPlay, chessMove::captureAndPromotion);
            }
            else {
                SIMPLE_MOVE(retval, brd, diagRight, coords, colorlessChessPiece::pawn, whoToPlay, chessMove::capture);
            }
        }
    }

    if ((int)coords.second - 1 >= 0) {
        //diagonal to the left
        auto diagLeft = boardCoords(coords.first + up, coords.second - 1);
        auto leftPieceColor = getPieceColor(brd[diagLeft.first][diagLeft.second]);
        if (leftPieceColor != whoToPlay && leftPieceColor != player::neither) {
            if (diagLeft.first == 0 || diagLeft.first == 7) {
                PAWN_PROMOTION(retval, brd, diagLeft, coords, whoToPlay, chessMove::captureAndPromotion);
            }
            else {
                SIMPLE_MOVE(retval, brd, diagLeft, coords, colorlessChessPiece::pawn, whoToPlay, chessMove::capture);
            }
        }
    }


    auto brdWithoutCapture = brd;
    brdWithoutCapture[previousMove.whereTo.first][previousMove.whereTo.second] = chessPiece::empty;
    //we do en passant
    if (previousMove.moveType == chessMove::doublePawn) {
        auto behindLastDoublePawnMove = boardCoords(previousMove.whereTo.first + up, previousMove.whereTo.second);
        //left
        if (previousMove.whereTo.second + 1 <= 7) {

            if (previousMove.whereTo.first == coords.first && previousMove.whereTo.second + 1 == coords.second) {


                SIMPLE_MOVE(retval, brdWithoutCapture, behindLastDoublePawnMove, coords, colorlessChessPiece::pawn, whoToPlay, chessMove::enPassant);
            }

        }
        //right
        if (previousMove.whereTo.second - 1 >= 0) {
            if (previousMove.whereTo.first == coords.first && previousMove.whereTo.second - 1 == coords.second) {
                SIMPLE_MOVE(retval, brdWithoutCapture, behindLastDoublePawnMove, coords, colorlessChessPiece::pawn, whoToPlay, chessMove::enPassant);
            }
        }
    }


    return retval;
}




#define BOUNDS_CHECK(coordsAt, xMult, yMult, acc) if ((int)coordsAt.first +acc*xMult < 0 || (int)coordsAt.second +acc*yMult < 0 || (int)coordsAt.first +i*xMult > 7 || (int)coordsAt.second +acc*yMult > 7) { break;}


std::vector<boardAndPreviousMove> getBishopMoves(board brd, boardCoords coords, player whoToPlay, bool requestingQueen = false) {
    std::vector<boardAndPreviousMove> moves;
    brd[coords.first][coords.second] = chessPiece::empty;
    //std::cout << "asd" << std::endl;

    colorlessChessPiece piecePlaying = requestingQueen ? colorlessChessPiece::queen : colorlessChessPiece::bishop;

    for (int x = -1; x < 2; x += 2)
    {
        for (int y = -1; y < 2; y += 2) {

            for (size_t i = 1; i < 8; i++)
            {
                //bounds checking
                BOUNDS_CHECK(coords, x, y, i);

                //next move
                boardCoords toMove = boardCoords(coords.first + i * x, coords.second + i * y);

                if (brd[toMove.first][toMove.second] == chessPiece::empty) {
                    SIMPLE_MOVE(moves, brd, toMove, coords, piecePlaying, whoToPlay, chessMove::normal);
                }
                else if (getPieceColor(brd[toMove.first][toMove.second]) == flipColor(whoToPlay)) {
                    SIMPLE_MOVE(moves, brd, toMove, coords, piecePlaying, whoToPlay, chessMove::capture);
                    break;
                }
                else {
                    break;
                }

            }
        }
    }


    return moves;
}

std::vector<boardAndPreviousMove> getRookMoves(board brd, boardCoords coords, player whoToPlay, bool requestingQueen = false) {
    std::vector<boardAndPreviousMove> moves;
    brd[coords.first][coords.second] = chessPiece::empty;

    colorlessChessPiece piecePlaying = requestingQueen ? colorlessChessPiece::queen : colorlessChessPiece::rook;

    for (size_t i = 0; i < 4; i++)
    {
        int x, y;
        if (i == 0) {
            x = 0;
            y = 1;
        }
        else if (i == 1) {
            x = -1;
            y = 0;
        }
        else if (i == 2) {
            x = 1;
            y = 0;
        }
        else if (i == 3) {
            x = 0;
            y = -1;
        }

        for (size_t i = 1; i < 8; i++)
        {
            //bounds checking
            BOUNDS_CHECK(coords, x, y, i);

            //next move
            boardCoords toMove = boardCoords(coords.first +  i * x, coords.second + i * y);


            if (brd[toMove.first][toMove.second] == chessPiece::empty) {
                SIMPLE_MOVE(moves, brd, toMove, coords, piecePlaying, whoToPlay, chessMove::normal);
            }
            else if (getPieceColor(brd[toMove.first][toMove.second]) != whoToPlay) {
                SIMPLE_MOVE(moves, brd, toMove, coords, piecePlaying, whoToPlay, chessMove::capture);
                break;
            }
            else {
                break;
            }
        }
    }

    return moves;
}


std::vector<boardAndPreviousMove> getKnightMoves(board brd, boardCoords coords, player whoToPlay) {
    std::vector<boardAndPreviousMove> moves;
    brd[coords.first][coords.second] = chessPiece::empty;

    for (size_t i = 0; i < 8; i++)
    {

        int x = coords.first, y = coords.second;
        switch (i)
        {
        case 0:
            x -= 2;
            y -= 1;
            break;
        case 1:
            x -= 1;
            y -= 2;
            break;
        case 2:
            x += 1;
            y -= 2;
            break;
        case 3:
            x += 2;
            y -= 1;
            break;
        case 4:
            x += 2;
            y += 1;
            break;
        case 5:
            x += 1;
            y += 2;
            break;
        case 6:
            x -= 1;
            y += 2;
            break;
        case 7:
            x -= 2;
            y += 1;
            break;
        default:
            break;
        }


        if (x > 7 || x < 0 || y > 7 || y < 0) {
            continue;
        }


        boardCoords toMove = boardCoords(x, y);

        if (brd[x][y] == chessPiece::empty) {
            SIMPLE_MOVE(moves, brd, toMove, coords, colorlessChessPiece::knight, whoToPlay, chessMove::normal);
        }
        else if (getPieceColor(brd[x][y]) == flipColor(whoToPlay)) {
            SIMPLE_MOVE(moves, brd, toMove, coords, colorlessChessPiece::knight, whoToPlay, chessMove::capture);
        }

    }
    return moves;
}


std::vector<boardAndPreviousMove> getQueenMoves(board brd, boardCoords coords, player whoToPlay) {
    std::vector<boardAndPreviousMove> moves;
    brd[coords.first][coords.second] = chessPiece::empty;

    moves = getBishopMoves(brd, coords, whoToPlay, true);

    auto rookMoves = getRookMoves(brd, coords, whoToPlay, true);

    moves.insert(moves.begin(), rookMoves.begin(), rookMoves.end());

    return moves;
}

std::vector<boardAndPreviousMove> getKingMoves(board brd, boardCoords coords, player whoToPlay) {
    std::vector<boardAndPreviousMove> moves;
    brd[coords.first][coords.second] = chessPiece::empty;



    for (int y = -1; y < 2; y++)
    {
        for (int x = -1; x < 2; x++)
        {
            int newY = coords.first + y, newX = coords.second + x;
            //bounds checking
            if (newY > 7 || newY < 0 || newX > 7 || newX < 0 || (x == 0 && y == 0)) {
                continue;
            }


            //
            boardCoords toMove = boardCoords(newY, newX);

            if (brd[toMove.first][toMove.second] == chessPiece::empty) {
                SIMPLE_MOVE(moves, brd, toMove, coords, colorlessChessPiece::king, whoToPlay, chessMove::normal);
            }
            else if (getPieceColor(brd[coords.first + y][coords.second + x]) == flipColor(whoToPlay)) {
                SIMPLE_MOVE(moves, brd, toMove, coords, colorlessChessPiece::king, whoToPlay, chessMove::capture);
            }
        }
    }


    return moves;
}

std::vector<boardAndPreviousMove> ChessGame::m_getPossibleMovesForBoard(const boardAndPreviousMove& brdAndMove, player whoToPlay)const
{
    //calculate all posible boards

    std::vector<boardAndPreviousMove> allMoves;


    //calculate all possible castling moves based on state remembered from previous turns
    if (whoToPlay == player::white) {
        if (m_WCanCastleLong) {
            auto newBoard = brdAndMove.first;
            newBoard[7][4] = chessPiece::empty;
            newBoard[7][0] = chessPiece::empty;
            newBoard[7][2] = chessPiece::whiteKing;
            newBoard[7][3] = chessPiece::whiteRook;
            chessMove newMove = chessMove(chessMove::moveTypes::castle, boardCoords(7, 4), boardCoords(7, 2), player::white, colorlessChessPiece::king);
            allMoves.push_back(boardAndPreviousMove(newBoard, newMove));
        }
        if (m_WCanCastleShort) {
            auto newBoard = brdAndMove.first;
            newBoard[7][4] = chessPiece::empty;
            newBoard[7][7] = chessPiece::empty;
            newBoard[7][6] = chessPiece::whiteKing;
            newBoard[7][5] = chessPiece::whiteRook;
            chessMove newMove = chessMove(chessMove::moveTypes::castle, boardCoords(7, 4), boardCoords(7, 6), player::white, colorlessChessPiece::king);
            allMoves.push_back(boardAndPreviousMove(newBoard, newMove));
        }
    }
    else if (whoToPlay == player::black) {
        if (m_BCanCastleLong) {
            auto newBoard = brdAndMove.first;
            newBoard[0][4] = chessPiece::empty;
            newBoard[0][0] = chessPiece::empty;
            newBoard[0][2] = chessPiece::blackKing;
            newBoard[0][3] = chessPiece::blackRook;
            chessMove newMove = chessMove(chessMove::moveTypes::castle, boardCoords(0, 4), boardCoords(0, 2), player::black, colorlessChessPiece::king);
            allMoves.push_back(boardAndPreviousMove(newBoard, newMove));
        }
        if (m_BCanCastleShort) {
            auto newBoard = brdAndMove.first;
            newBoard[0][4] = chessPiece::empty;
            newBoard[0][7] = chessPiece::empty;
            newBoard[0][6] = chessPiece::blackKing;
            newBoard[0][5] = chessPiece::blackRook;
            chessMove newMove = chessMove(chessMove::moveTypes::castle, boardCoords(0, 4), boardCoords(0, 6), player::black, colorlessChessPiece::king);
            allMoves.push_back(boardAndPreviousMove(newBoard, newMove));
        }
    }


    for (size_t i = 0; i < 8; i++)
    {
        for (size_t o = 0; o < 8; o++)
        {
            auto piece = brdAndMove.first[i][o];

            if (piece != chessPiece::empty && getPieceColor(piece) == whoToPlay) {



                auto colorlessPiece = makeColorlessPiece(piece);
                std::vector<boardAndPreviousMove> pieceMoves;
                switch (colorlessPiece)
                {
                case colorlessChessPiece::pawn:
                    pieceMoves = getPawnMoves(brdAndMove.first, brdAndMove.second, boardCoords(i, o), whoToPlay);

                    break;
                case colorlessChessPiece::bishop:
                    pieceMoves = getBishopMoves(brdAndMove.first, boardCoords(i, o), whoToPlay);
                    break;
                case colorlessChessPiece::rook:
                    pieceMoves = getRookMoves(brdAndMove.first, boardCoords(i, o), whoToPlay);
                    break;
                case colorlessChessPiece::knight:
                    pieceMoves = getKnightMoves(brdAndMove.first, boardCoords(i, o), whoToPlay);
                    break;
                case colorlessChessPiece::queen:
                    pieceMoves = getQueenMoves(brdAndMove.first, boardCoords(i, o), whoToPlay);
                    break;
                case colorlessChessPiece::king:
                    pieceMoves = getKingMoves(brdAndMove.first, boardCoords(i, o), whoToPlay);
                    break;
                }


                allMoves.insert(allMoves.end(), pieceMoves.begin(), pieceMoves.end());
            }



        }
    }

    return allMoves;
}

bool ChessGame::getRepeatedPosition() const noexcept
{
    return m_repeatedPosition;
}

size_t ChessGame::getMovesWithoutCaptureOrPawnMove() const noexcept
{
    return m_movesWithoutCaptureOrPawnMove;
}

ChessGame::ChessGame(const std::string& fen)
{
    for (size_t i = 0; i < 8; i++)
    {
        for (size_t o = 0; o < 8; o++)
        {
            m_current[i][o] = chessPiece::empty;
        }
    }
    size_t row = 0;
    size_t column = 0;
    size_t chars = 0;
    for (char c:fen) {
        chars++;
        switch (c) {
        case '/':
            row++;
            column = 0;
            break;
        case ' ':
            goto afterLoop;
        case 'r':
            m_current[row][column] = chessPiece::blackRook;
            column++;
            break;
        case 'n':
            m_current[row][column] = chessPiece::blackKnight;
            column++;
            break;
        case 'b':
            m_current[row][column] = chessPiece::blackBishop;
            column++;
            break;
        case 'q':
            m_current[row][column] = chessPiece::blackQueen;
            column++;
            break;
        case 'k':
            m_current[row][column] = chessPiece::blackKing;
            column++;
            break;
        case 'p':
            m_current[row][column] = chessPiece::blackPawn;
            column++;
            break;
        case 'R':
            m_current[row][column] = chessPiece::whiteRook;
            column++;
            break;
        case 'N':
            m_current[row][column] = chessPiece::whiteKnight;
            column++;
            break;
        case 'B':
            m_current[row][column] = chessPiece::whiteBishop;
            column++;
            break;
        case 'Q':
            m_current[row][column] = chessPiece::whiteQueen;
            column++;
            break;
        case 'K':
            m_current[row][column] = chessPiece::whiteKing;
            column++;
            break;
        case 'P':
            m_current[row][column] = chessPiece::whitePawn;
            column++;
            break;
        default:
            column += c - '0';
            break;
        } 
    }
afterLoop:
    if (fen[chars] == 'w') {
        m_whoToPlay = player::white;
    }
    else {
        m_whoToPlay = player::black;
    }

    m_BHasLongCastlingRights = false;
    m_BHasShortCastlingRights = false;
    m_WHasLongCastlingRights = false;
    m_WHasShortCastlingRights = false;
    std::string enPassantSquare = "";
    for (size_t i = chars+2; chars < fen.size(); chars++) {
        switch (fen[i]) {
        case ' ':
        case '-':
            break;
        case 'K':
            m_WHasShortCastlingRights = true;
            break;
        case 'Q':
            m_WHasLongCastlingRights = true;
            break;
        case 'q':
            m_BHasLongCastlingRights = true;
            break;
        case 'k':
            m_BHasShortCastlingRights = true;
            break;
        default:
            enPassantSquare.push_back(fen[i]);
            break;
        }
    }
    //add a double pawn move as the last move for the opposite player whose turn it is. This allows for en passant to be considered as a valid move.
    if (!enPassantSquare.empty()) {
        int pCol = enPassantSquare[0] - 'a';
        int pRow = 7-(enPassantSquare[1] - '1');
        if (pRow == 2) {
            m_moves.push_back(chessMove(chessMove::moveTypes::doublePawn,boardCoords(pRow - 1,pCol), boardCoords(pRow + 1, pCol), player::black,colorlessChessPiece::pawn));
        }
        else {
            m_moves.push_back(chessMove(chessMove::moveTypes::doublePawn, boardCoords(pRow + 1, pCol), boardCoords(pRow - 1, pCol), player::white, colorlessChessPiece::pawn));
        }
    }
}

ChessGame::ChessGame()
{
    //make starting board
    for (size_t i = 0; i < 8; i++)
    {
        for (size_t o = 0; o < 8; o++)
        {
            m_current[i][o] = chessPiece::empty;
        }
    }

    m_current[0][0] = chessPiece::blackRook;
    m_current[0][1] = chessPiece::blackKnight;
    m_current[0][2] = chessPiece::blackBishop;
    m_current[0][3] = chessPiece::blackQueen;
    m_current[0][4] = chessPiece::blackKing;
    m_current[0][5] = chessPiece::blackBishop;
    m_current[0][6] = chessPiece::blackKnight;
    m_current[0][7] = chessPiece::blackRook;
    for (size_t i = 0; i < 8; i++)
    {
        m_current[1][i] = chessPiece::blackPawn;
    }


    m_current[7][0] = chessPiece::whiteRook;
    m_current[7][1] = chessPiece::whiteKnight;
    m_current[7][2] = chessPiece::whiteBishop;
    m_current[7][3] = chessPiece::whiteQueen;
    m_current[7][4] = chessPiece::whiteKing;
    m_current[7][5] = chessPiece::whiteBishop;
    m_current[7][6] = chessPiece::whiteKnight;
    m_current[7][7] = chessPiece::whiteRook;

    for (size_t i = 0; i < 8; i++)
    {
        m_current[6][i] = chessPiece::whitePawn;
    }

}


bool ChessGame::m_checkWouldCaptureKing(const boardAndPreviousMove &brd)const
{
    auto boardsAndMove = m_getPossibleMovesForBoard(brd, flipColor(m_whoToPlay));
    for (auto x : boardsAndMove) {
        bool hasKing = false;
        for (auto col : x.first) {
            for (auto elem : col) {
                if (elem == makePiece(colorlessChessPiece::king, m_whoToPlay)) {
                    hasKing = true;
                    break;
                }
            }
            if (hasKing){
                break;
            }
        }
        if (!hasKing) {
            return true;
        }
    }

    return false;
}

void ChessGame::m_setCanCastle(const boardAndPreviousMove& brdMove, player whoToPlay)
{

    if (brdMove.first[7][1] == chessPiece::empty && brdMove.first[7][2] == chessPiece::empty || brdMove.first[7][3] == chessPiece::empty) {
        m_WCanCastleLong = true;
    }
    else {
        m_WCanCastleLong = false;
    }
    if (brdMove.first[7][5] == chessPiece::empty && brdMove.first[7][6] == chessPiece::empty) {
        m_WCanCastleShort = true;
    }
    else {
        m_WCanCastleShort = false;
    }
    if (brdMove.first[0][1] == chessPiece::empty && brdMove.first[0][2] == chessPiece::empty || brdMove.first[0][3] == chessPiece::empty) {
        m_BCanCastleLong = true;
    }
    else {
        m_BCanCastleLong = false;
    }
    if (brdMove.first[0][5] == chessPiece::empty && brdMove.first[0][6] == chessPiece::empty) {
        m_BCanCastleShort = true;
    }
    else {
        m_BCanCastleShort = false;
    }

    if (!m_WHasLongCastlingRights) {
        m_WCanCastleLong = false;
    }
    if (!m_WHasShortCastlingRights) {
        m_WCanCastleShort = false;
    }

    if (!m_BHasLongCastlingRights) {
        m_BCanCastleLong = false;
    }
    if (!m_BHasShortCastlingRights) {
        m_BCanCastleShort = false;
    }


    //check if king ever moved
    if (brdMove.second.initalPiece == colorlessChessPiece::king) {
        if (m_whoToPlay == player::white) {
            m_WMovedKing = true;
        }
        if (m_whoToPlay == player::black) {
            m_BMovedKing = true;
        }
    }
    //check if rook ever moved
    if (brdMove.second.initalPiece == colorlessChessPiece::rook) {
        if (m_whoToPlay == player::white) {
            if (brdMove.second.whereFrom == boardCoords(7, 0)) {
                m_WMovedLongRook = true;
            }
            else if (brdMove.second.whereFrom == boardCoords(7, 7)) {
                m_WMovedShortRook = true;
            }
        }
        if (m_whoToPlay == player::black) {
            if (brdMove.second.whereFrom == boardCoords(0, 0)) {
                m_BMovedLongRook = true;
            }
            else if (brdMove.second.whereFrom == boardCoords(0, 7)) {
                m_BMovedShortRook = true;
            }
        }
    }

    //cannot castle if either rook or king moved
    if (m_WMovedKing || m_WMovedLongRook) {
        m_WCanCastleLong = false;
    }
    if (m_WMovedKing || m_WMovedShortRook) {
        m_WCanCastleShort = false;
    }
    if (m_BMovedKing || m_BMovedLongRook) {
        m_BCanCastleLong = false;
    }
    if (m_BMovedKing || m_BMovedShortRook) {
        m_BCanCastleShort = false;
    }

    //cannot castle if rook has been captured
    if (brdMove.first[7][0] != chessPiece::whiteRook) {
        m_WCanCastleLong = false;
    }
    if (brdMove.first[7][7] != chessPiece::whiteRook) {
        m_WCanCastleShort = false;
    }
    if (brdMove.first[0][0] != chessPiece::blackRook) {
        m_BCanCastleLong = false;
    }
    if (brdMove.first[0][7] != chessPiece::blackRook) {
        m_BCanCastleShort = false;
    }

    //cannot castle along a check or while in a check
    for (size_t i = 0; i < 2; i++) {
        auto boardsAndMove = m_getPossibleMovesForBoard(brdMove, whoToPlay);
        if (whoToPlay == player::black) {
            for (auto x : boardsAndMove) {

                if (x.second.whereTo == boardCoords(7, 4) || x.second.whereTo == boardCoords(7, 3) || x.second.whereTo == boardCoords(7, 2)) {
                    m_WCanCastleLong = false;
                }
                if (x.second.whereTo == boardCoords(7, 4) || x.second.whereTo == boardCoords(7, 5) || x.second.whereTo == boardCoords(7, 6)) {
                    m_WCanCastleShort = false;
                }
            }
        }
        if (whoToPlay == player::white) {
            for (auto x : boardsAndMove) {

                if (x.second.whereTo == boardCoords(0, 4) || x.second.whereTo == boardCoords(0, 3) || x.second.whereTo == boardCoords(0, 2)) {
                    m_BCanCastleLong = false;
                }
                if (x.second.whereTo == boardCoords(0, 4) || x.second.whereTo == boardCoords(0, 5) || x.second.whereTo == boardCoords(0, 6)) {
                    m_BCanCastleShort = false;
                }
            }
        }
        whoToPlay = flipColor(whoToPlay);
    }



}


std::vector<boardAndPreviousMove> ChessGame::getPossibleBoards(gameCondition *condition)const
{

    auto allMoves = m_getPossibleMovesForBoard(boardAndPreviousMove{m_current,m_moves.back()}, m_whoToPlay);
    std::vector<boardAndPreviousMove> vettedMoves;

    //tie by no legal moves, but no king capture next turn
    if (allMoves.size() == 0) {
        if (condition != nullptr) {
            *condition = gameCondition::tieByStalemate;
        }
    }else{
        for (auto x : allMoves)
        {
            if (!m_checkWouldCaptureKing(x)) {
                vettedMoves.push_back(x);
            } 
        }

        //victory by checkmate
        if (vettedMoves.size() == 0) {
            if (condition != nullptr) {
                *condition = (m_whoToPlay == player::white ? gameCondition::blackVictory : gameCondition::whiteVictory);
            }

        }
        else if(m_movesWithoutCaptureOrPawnMove > 50) {
            if (condition != nullptr) {
                *condition = gameCondition::tieBy50Moves;
            }

        }
        else {
            if (condition != nullptr) {
                *condition = gameCondition::playing;
            }

        }
    }

    return vettedMoves;
}


bool boardsAreTheSame(const board & brd1, const board& brd2) {
    for (size_t i = 0; i < 8; i++)
    {
        for (size_t o = 0; o < 8; o++)
        {
            if (brd1[i][o] != brd2[i][o]) {
                return false;
            }
        }
    }
    return true;
}


void ChessGame::setNext(boardAndPreviousMove brdMove)
{

    m_moves.push_back(brdMove.second);
    m_pastBoards.push_back(m_current);
    m_current = brdMove.first;


    m_setCanCastle(brdMove, m_whoToPlay);



    if (brdMove.second.moveType != chessMove::moveTypes::capture && brdMove.second.initalPiece != colorlessChessPiece::pawn) {
        m_movesWithoutCaptureOrPawnMove++;
    }
    else {
        m_movesWithoutCaptureOrPawnMove = 0;
    }


    m_repeatedPosition = false;

    for (auto x: m_pastBoards) {
        if (boardsAreTheSame(x, m_current)) {
            m_repeatedPosition = true;
        }
    }

    m_whoToPlay = flipColor(m_whoToPlay);
    

}


board ChessGame::getCurrentBoard() const noexcept
{
    return m_current;
}

chessMove::chessMove(moveTypes p_moveType, boardCoords p_whereFrom, boardCoords p_whereTo, player p_who, colorlessChessPiece piece):
    moveType(p_moveType),
    whereFrom(p_whereFrom),
    whereTo(p_whereTo),
    who(p_who),
    initalPiece(piece)
{
}

chessMove::chessMove():
    moveType(notAMove)
{
}

static std::map<chessMove::moveTypes, std::string> moveTypesToString = {
    {chessMove::capture, "capture"},
    {chessMove::captureAndPromotion, "capture and promotion"},
    {chessMove::castle, "castle"},
    {chessMove::doublePawn, "double pawn"},
    {chessMove::enPassant, "en passant"},
    {chessMove::normal, "normal"},
    {chessMove::promotion, "promotion"}
};

static std::map<chessMove::promotionTypes, std::string> promotionTypesToString = {
    {chessMove::toBishop, "bishop"},
    {chessMove::toKnight, "knight"},
    {chessMove::toQueen, "queen"},
    {chessMove::toRook, "rook"}
};

static std::map<colorlessChessPiece, std::string> colorlessChessPieceToString = {
    {colorlessChessPiece::bishop, "bishop"},
    {colorlessChessPiece::knight, "knight"},
    {colorlessChessPiece::king, "king"},
    {colorlessChessPiece::pawn, "pawn"},
    {colorlessChessPiece::queen, "queen"},
    {colorlessChessPiece::rook, "rook"}
};

std::string getBoardCoordsString(boardCoords coords) {
    std::stringstream ss;
    //coordinates are flipped to match horizontal first vertical second
    ss << "(" << (int)coords.second<< ";" << (int)coords.first << ")";
    return ss.str();
}


std::string chessMove::getStringRepresentation() const
{
    std::stringstream ss;
    ss << "move type: " << moveTypesToString[moveType] << ", from: " << getBoardCoordsString(whereFrom) << ", to: " << getBoardCoordsString(whereTo) << ", by: " << (who==player::white?"white":"black") << ", piece: " << colorlessChessPieceToString[initalPiece];
    if (promotionTo != notPromotion) {
        ss << ", promotion to: " << promotionTypesToString[promotionTo];
    }



    return ss.str();
}

board flipBoard(const board& brd)
{
    //we invert the board
    board tempBoard;
    for (size_t rank = 0; rank < 8; rank++)
    {
        for (size_t file = 0; file < 8; file++)
        {
            tempBoard[rank][file] = brd[7 - rank][file];
        }
    }
    return tempBoard;
}

//we must provide the same numeric representation regardless of who asks for it. As if colors did not matter and we only needed to know who is friendly and enemy on our side of the board.
std::vector<float> getNumericRepresentationOfBoard(board brd, player whoToPlay)
{

    if (whoToPlay == player::white) {
        brd = flipBoard(brd);
    }
    

    //we multiply 8*8 which is the amount of squares, there are 6 pieces of two colors, and a posibility of an empty square. so 8*8*6*3
    //the first number tells wether square (0,0) is empty, the first 6 numbers correspond to the friendly pieces of square (0,0) the next
    // 6 numbers correspond to the enemy pieces.

    std::vector<float> retval;

    for (size_t rank = 0; rank < 8; rank++)
    {
        for (size_t file = 0; file < 8; file++)
        {
            retval.push_back(brd[rank][file] == chessPiece::empty);
            retval.push_back(brd[rank][file] == makePiece(colorlessChessPiece::bishop,whoToPlay));
            retval.push_back(brd[rank][file] == makePiece(colorlessChessPiece::king,whoToPlay));
            retval.push_back(brd[rank][file] == makePiece(colorlessChessPiece::knight,whoToPlay));
            retval.push_back(brd[rank][file] == makePiece(colorlessChessPiece::pawn,whoToPlay));
            retval.push_back(brd[rank][file] == makePiece(colorlessChessPiece::queen,whoToPlay));
            retval.push_back(brd[rank][file] == makePiece(colorlessChessPiece::rook,whoToPlay));
            retval.push_back(brd[rank][file] != makePiece(colorlessChessPiece::bishop, whoToPlay));
            retval.push_back(brd[rank][file] != makePiece(colorlessChessPiece::king, whoToPlay));
            retval.push_back(brd[rank][file] != makePiece(colorlessChessPiece::knight, whoToPlay));
            retval.push_back(brd[rank][file] != makePiece(colorlessChessPiece::pawn, whoToPlay));
            retval.push_back(brd[rank][file] != makePiece(colorlessChessPiece::queen, whoToPlay));
            retval.push_back(brd[rank][file] != makePiece(colorlessChessPiece::rook, whoToPlay));
        }
    }

    return retval;
}

void ChessGame::addExtraInputs(std::vector<float>& pastInputs, player whoIsPlayed)const 
{
    pastInputs.push_back(m_movesWithoutCaptureOrPawnMove/50);
    pastInputs.push_back(m_repeatedPosition?1:0);
    if (whoIsPlayed == player::white) {
        pastInputs.push_back(m_WCanCastleLong?1:0);
        pastInputs.push_back(m_WCanCastleShort?1:0);
        pastInputs.push_back(m_BCanCastleLong ? 1 : 0);
        pastInputs.push_back(m_BCanCastleShort ? 1 : 0);
    }
    else if(whoIsPlayed ==  player::black) {
        pastInputs.push_back(m_BCanCastleLong ? 1 : 0);
        pastInputs.push_back(m_BCanCastleShort ? 1 : 0);
        pastInputs.push_back(m_WCanCastleLong ? 1 : 0);
        pastInputs.push_back(m_WCanCastleShort ? 1 : 0);
    }
}
