#include "Board.h"
#include <vector>

#pragma once
class MCTS
{
public:
	Player player;
	int wins;
	int simulationsCount;
	bool visitedInCurrentIteration;
	Board board;
	MCTS *parent;
	std::vector<MCTS *> children;
	MCTS(MCTS *parent, Board board, Player player);
	void add_child(MCTS *child);
};

