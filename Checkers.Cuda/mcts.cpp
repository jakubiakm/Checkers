#include "mcts.h"


Mcts::Mcts() : number_of_total_simulations(0)
{
}


void Mcts::GenerateRoot(Board startBoard, int movesCount, Move* possibleMoves)
{
	MctsNode* root = new MctsNode(0, startBoard);
	for (int i = 0; i != movesCount; i++)
	{
		MctsNode* child = new MctsNode(root, startBoard.GetBoardAfterMove(possibleMoves[i]));
		root->AddChild(child);
	}
	Mcts::root = root;
}

__host__ MctsNode* Mcts::SelectNode(MctsNode *parent)
{
	MctsNode *leafNode = parent;
	while (leafNode->children.size() != 0)
	{
		double max = 0;
		int ind;
		bool visited = false;
		for (int i = 0; i != leafNode->children.size(); i++)
		{
			if (leafNode->children[i]->visited_in_current_iteration)
				continue;
			if (leafNode->children[i]->simulations_count == 0)
			{
				ind = i;
				visited = true;
				break;
			}
			if (leafNode->children[i]->wins / leafNode->children[i]->simulations_count + UCT_CONSTANT * sqrt(log(number_of_total_simulations) / leafNode->children[i]->simulations_count) > max)
			{
				max = leafNode->children[i]->wins / leafNode->children[i]->simulations_count + UCT_CONSTANT * sqrt(log(number_of_total_simulations) / leafNode->children[i]->simulations_count);
				ind = i;
				visited = true;
			}
		}
		if (!visited)
			return 0;
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
			return 0;
		auto moves = leafNode->board.GetPossibleMovesCpu(moves_count);
		if (moves_count == 0)
			return 0;
		for (int i = 0; i != moves_count; i++)
		{
			leafNode->AddChild(new MctsNode(leafNode, leafNode->board.GetBoardAfterMove(moves[i])));
		}

		return leafNode->children[0];
	}
}

void Mcts::BackpropagateSimulations(MctsNode *leaf, int duplication_count)
{
	number_of_total_simulations++;
	leaf->visited_in_current_iteration = true;
	while (leaf != 0)
	{
		leaf->simulations_count += duplication_count;
		leaf = leaf->parent;
	}
}

void Mcts::BackpropagateResults(std::vector<MctsNode*> vector, int *results)
{
	for (int i = 0; i != vector.size(); i++)
	{
		MctsNode *leaf = vector[i];
		while (leaf != 0)
		{
			leaf->wins += results[i];
			leaf->visited_in_current_iteration = 0;
			leaf = leaf->parent;
		}
	}
}

int Mcts::GetBestMove()
{
	double max = 0;
	int ind = 0;
	for (int i = 0; i != root->children.size(); i++)
	{
		if (root->children[i]->visited_in_current_iteration)
			continue;
		if (root->children[i]->simulations_count == 0)
		{
			ind = i;
			break;
		}
		if (root->children[i]->wins / root->children[i]->simulations_count + UCT_CONSTANT * sqrt(log(number_of_total_simulations) / root->children[i]->simulations_count) > max)
		{
			max = root->children[i]->wins / root->children[i]->simulations_count + UCT_CONSTANT * sqrt(log(number_of_total_simulations) / root->children[i]->simulations_count);
			ind = i;
		}
	}
	return ind;
}