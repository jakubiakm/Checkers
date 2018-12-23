#pragma once
#include <vector>
#include "board.cuh"

class MctsNode
{
public:
	Player player;
	int wins;
	int simulations_count;
	bool visited_in_current_iteration;
	Board board;
	MctsNode *parent;
	std::vector<MctsNode *> children;

	MctsNode(MctsNode *parent, Board board);
	void AddChild(MctsNode * child);
};
