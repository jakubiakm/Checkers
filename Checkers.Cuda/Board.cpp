#include "Board.h"

Board::Board()
{

}

Board::Board(int size, int* pieces, Player player) : size(size), player(player)
{
	Board::pieces = new int[size * size];
	for (int i = 0; i != size * size; i++)
		Board::pieces[i] = pieces[i];
}

Board::~Board()
{
	//delete[]Pieces;
}

std::vector<Move> Board::get_possible_moves()
{
	std::vector<Move> possibleMoves;
	return possibleMoves;
}