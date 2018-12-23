#include "mcts.h"


Mcts::Mcts() : number_of_total_simulations(0)
{
}


MctsNode* Mcts::GenerateRoot(Board startBoard, int movesCount, Move* possibleMoves)
{
	MctsNode* root = new MctsNode(NULL, startBoard);
	for (int i = 0; i != movesCount; i++)
	{
		MctsNode* child = new MctsNode(root, startBoard.GetBoardAfterMove(possibleMoves[i]));
		root->AddChild(child);
	}
	return root;
}

MctsNode* Mcts::SelectNode(MctsNode *parent)
{
	MctsNode *leafNode = parent;
	while (leafNode->children.size() != 0)
	{
		int max = 0;
		int ind;
		for (int i = 0; i != leafNode->children.size(); i++)
		{
			if (leafNode->children[i]->simulations_count == 0)
			{
				ind = i;
				break;
			}
			if (leafNode->children[i]->wins / leafNode->children[i]->simulations_count + UCT_CONSTANT * sqrt(log(number_of_total_simulations) / leafNode->children[i]->simulations_count) > max)
			{
				max = leafNode->children[i]->wins / leafNode->children[i]->simulations_count + UCT_CONSTANT * sqrt(log(number_of_total_simulations) / leafNode->children[i]->simulations_count);
				ind = i;
			}
		}
		leafNode = leafNode->children[ind];
	}
	if (leafNode->simulations_count == 0)
	{
		return leafNode;
	}
	else
	{
		int moves_count = 0;
		if (leafNode->visited_in_current_iteration)
			return NULL;
		auto moves = leafNode->board.GetPossibleMoves(moves_count);
		if (moves_count == 0)
			return NULL;
		for (int i = 0; i != moves_count; i++)
		{
			leafNode->AddChild(new MctsNode(leafNode, leafNode->board.GetBoardAfterMove(moves[i])));
		}

		return leafNode->children[0];
	}
}

void Mcts::BackpropagateSimulations(MctsNode *leaf)
{
	number_of_total_simulations++;
	while (leaf != NULL)
	{
		leaf->simulations_count++;
		leaf = leaf->parent;
	}
}