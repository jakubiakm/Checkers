#pragma once

#include "mctsnode.h"

class Mcts
{
public:
	MctsNode* root;

	Mcts();
	void GenerateRoot(Board startBoard, int movesCount, Move* possibleMoves);
	MctsNode* SelectNode(MctsNode *parent);
	void BackpropagateSimulations(MctsNode *leaf, int duplication_count);
	void BackpropagateResults(std::vector<MctsNode*> vector, int *results);
	int GetBestMove();
private:
	double number_of_total_simulations;

	// Sta�a w algorytmiu UCT
	const double UCT_CONSTANT = 2;
};

