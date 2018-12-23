#pragma once

#include "mctsnode.h"

class Mcts
{
public:
	MctsNode* root;

	Mcts();
	void GenerateRoot(Board startBoard, int movesCount, Move* possibleMoves);
	MctsNode* SelectNode(MctsNode *parent);
	void BackpropagateSimulations(MctsNode *leaf);
private:
	int number_of_total_simulations;

	// Sta³a w algorytmiu UCT
	const int UCT_CONSTANT = 2;
};

