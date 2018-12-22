#include "Board.h"

Board::Board()
{

}

Board::Board(int size, int* pieces) : size(size)
{
	Board::pieces = new int[size * size];
	for (int i = 0; i != size * size; i++)
		Board::pieces[i] = pieces[i];
}

Board::~Board()
{
	//delete[]Pieces;
}
