#include "Board.h"



Board::Board(int size, int* pieces)
{
	Size = size;
	Pieces = new int[size * size];
	for (int i = 0; i != Size * Size; i++)
		Pieces[i] = pieces[i];
}

Board::~Board()
{
	delete[]Pieces;
}
