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
	MCTS::MCTS(MCTS *parent, Board board) : parent(parent), board(board), wins(0), simulationsCount(0), visitedInCurrentIteration(false)
	{
	}
	void MCTS::add_child(MCTS * child)
	{
		children.push_back(child);
	}
};