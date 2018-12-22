#include "Board.h"
#include <vector>

#pragma once
class MCTS
{
public:
	int player;
	int wins;
	int simulationsCount;
	Board board;
	MCTS *parent;
	std::vector<MCTS *> children;
	MCTS(MCTS *parent, Board board, int player);
	void add_child(MCTS *child);
};

