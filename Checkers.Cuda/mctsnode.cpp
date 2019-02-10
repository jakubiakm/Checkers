

#include "mctsnode.h"


MctsNode::MctsNode(MctsNode *parent, Board board) : parent(parent), board(board), wins(0), simulations_count(0), visited_in_current_iteration(false)
{
}

void MctsNode::AddChild(MctsNode * child)
{
	children.push_back(child);
}
