#include "MCTS.h"


MCTS::MCTS(MCTS *parent, Board board, int player) : parent(parent), board(board), player(player), wins(0), simulationsCount(0)
{
}

void MCTS::add_child(MCTS * child)
{
	children.push_back(child);
}
