#pragma once
class Board
{
public:
	int Size;
	int* Pieces;
	Board(int size, int* pieces);
	Board();
	~Board();
};

