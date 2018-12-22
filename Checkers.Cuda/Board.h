#pragma once
enum Player
{
	White, Black
};
class Board
{
public:
	int size;
	int* pieces;
	Board(int size, int* pieces);
	Board();
	~Board();
};

