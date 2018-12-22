#include <vector>
#include "Move.h"

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
	Board(int size, int* pieces, Player player);
	Board();
	~Board();
	Player player;
	std::vector<Move> get_possible_moves();
};

