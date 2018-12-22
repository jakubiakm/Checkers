
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h> 
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Move.cuh"

#define cConst 2

int N = 0;


enum Player
{
	White, Black
};
class Board
{
public:
	int size;
	int* pieces;
	Player player;
	__device__ __host__ Board(int size, int* _pieces, Player player) : size(size), player(player)
	{
		pieces = new int[size * size];
		for (int i = 0; i != size * size; i++)
		{
			pieces[i] = _pieces[i];
		}
	}
	__device__ __host__ Board();
	__device__ __host__ ~Board();
	__device__ __host__ Move* Board::get_possible_moves(int &moves_count);
	__device__ __host__ Board Board::get_board_after_move(Move move);
};

__device__ __host__ Board::Board()
{

}

__device__ __host__ Board::~Board()
{
	//delete[]Pieces;
}

__device__ __host__ Move* Board::get_possible_moves(int &moves_count)
{
	moves_count = 0;
	Move* possibleMoves = new Move[100];
	return possibleMoves;
}

__device__ __host__ Board Board::get_board_after_move(Move move)
{
	int *_pieces = new int[size * size];
	for (int i = 0; i != size * size; i++)
	{
		_pieces[i] = pieces[i];
	}
	for (int i = 0; i != move.beatedPiecesCount; i++)
	{
		_pieces[move.beatedPieces[i]] = 0;
	}
	_pieces[move.newPosition] = _pieces[move.oldPosition];
	_pieces[move.oldPosition] = 0;
	return Board(size, _pieces, player == Player::White ? Player::Black : Player::White);
}

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
	MCTS(MCTS *parent, Board board);
	void add_child(MCTS *child);
};

MCTS::MCTS(MCTS *parent, Board board) : parent(parent), board(board), wins(0), simulationsCount(0), visitedInCurrentIteration(false)
{
}

void MCTS::add_child(MCTS * child)
{
	children.push_back(child);
}


MCTS* GenerateRoot(Board startBoard, int movesCount, Move* possibleMoves)
{
	MCTS* root = new MCTS(NULL, startBoard);
	for (int i = 0; i != movesCount; i++)
	{
		MCTS* child = new MCTS(root, startBoard.get_board_after_move(possibleMoves[i]));
		root->add_child(child);
	}
	return root;
}

MCTS* SelectNode(MCTS *parent)
{
	MCTS *leafNode = parent;
	while (leafNode->children.size() != 0)
	{
		int max = 0;
		int ind;
		for (int i = 0; i != leafNode->children.size(); i++)
		{
			if (leafNode->children[i]->simulationsCount == 0)
			{
				ind = i;
				break;
			}
			if (leafNode->children[i]->wins / leafNode->children[i]->simulationsCount + cConst * sqrt(log(N) / leafNode->children[i]->simulationsCount) > max)
			{
				max = leafNode->children[i]->wins / leafNode->children[i]->simulationsCount + cConst * sqrt(log(N) / leafNode->children[i]->simulationsCount);
				ind = i;
			}
		}
		leafNode = leafNode->children[ind];
	}
	if (leafNode->simulationsCount == 0)
	{
		return leafNode;
	}
	else
	{
		int moves_count = 0;
		if (leafNode->visitedInCurrentIteration)
			return NULL;
		auto moves = leafNode->board.get_possible_moves(moves_count);
		if (moves_count == 0)
			return NULL;
		for (int i = 0; i != moves_count; i++)
		{
			leafNode->add_child(new MCTS(leafNode, leafNode->board.get_board_after_move(moves[i])));
		}

		return leafNode->children[0];
	}
}

void BackpropagateSimulations(MCTS *leaf)
{
	N++;
	while (leaf != NULL)
	{
		leaf->simulationsCount++;
		leaf = leaf->parent;
	}
}

__device__ int RolloutBoard(Board board)
{
	return 1;
}

__global__ void RolloutGames(Board* rolloutBoardsVector, int* results, int size)
{
	const long numThreads = blockDim.x * gridDim.x;
	const long threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (long long ind = threadID; ind < size; ind += numThreads)
	{
		results[ind] = RolloutBoard(rolloutBoardsVector[ind]);
	}
}

// internal variable (example, not really necessary here)
static volatile int PRINT_ERRORS = 1;	// true

// cuda helper function (internal)
int checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		if (PRINT_ERRORS)
			printf("Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		return err;
	}
	return 0; // cudaSuccess
}

extern "C" int __declspec(dllexport) __stdcall MakeMoveCpu
(
	int boardSize,
	int _player, //0 - bia造, 1 - czarny
	int* board, //0 - puste, 1 - bia造 pion, 2 - bia豉 dama, 3 - czarny pion, 4 - czarna dama
	int* possibleMoves
)
{
	Board startBoard = Board(boardSize, board, _player == 0 ? Player::White : Player::Black);
	int possibleMovesCount = possibleMoves[0];
	int ind = 1;
	Move* moves = new Move[possibleMovesCount];
	for (int i = 0; i != possibleMovesCount; i++)
	{
		int beatedPiecesCount = possibleMoves[ind++];
		int *beatedPieces = new int[beatedPiecesCount];
		for (int j = 0; j != beatedPiecesCount; j++)
		{
			beatedPieces[j] = possibleMoves[ind++];
		}
		moves[i] = Move(
			possibleMoves[ind++],
			possibleMoves[ind++],
			beatedPiecesCount,
			beatedPieces
		);
	}
	return possibleMovesCount - 1;
}

extern "C" int __declspec(dllexport) __stdcall MakeMoveGpu
(
	int boardSize,
	int _player, //0 - bia造, 1 - czarny
	int* board, //0 - puste, 1 - bia造 pion, 2 - bia豉 dama, 3 - czarny pion, 4 - czarna dama
	int* possibleMoves
)
{
	Player player = _player == 0 ? Player::White : Player::Black;
	Board startBoard = Board(boardSize, board, player);
	int possibleMovesCount = possibleMoves[0];
	int ind = 1;
	Move* moves = new Move[possibleMovesCount];
	for (int i = 0; i != possibleMovesCount; i++)
	{
		int beatedPiecesCount = possibleMoves[ind++];
		int *beatedPieces = new int[beatedPiecesCount];
		for (int j = 0; j != beatedPiecesCount; j++)
		{
			beatedPieces[j] = possibleMoves[ind++];
		}
		moves[i] = Move(
			possibleMoves[ind++],
			possibleMoves[ind++],
			beatedPiecesCount,
			beatedPieces
		);
	}

	MCTS* root = GenerateRoot(startBoard, possibleMovesCount, moves);
	std::vector<MCTS*> rolloutVector;

	int tmp = PRINT_ERRORS;
	int cuerr;
	int blockSize = 1024;      // The launch configurator returned block size 
	int gridSize = 1024;       // The actual grid size needed, based on input size 

	for (int i = 0; i != blockSize * gridSize; i++)
	{
		MCTS* node = SelectNode(root);
		if (node == NULL || node->visitedInCurrentIteration)
			break;
		BackpropagateSimulations(node);
		rolloutVector.push_back(node);
	}

	Board* boards_to_rollout = new Board[rolloutVector.size()];
	Board* boards_d;
	int
		*results_d,
		*results = new int[rolloutVector.size()];

	for (int i = 0; i != rolloutVector.size(); i++)
	{
		boards_to_rollout[i] = rolloutVector[i]->board;
	}

	cudaMalloc((void**)&boards_d, rolloutVector.size() * sizeof(Board));
	cudaMalloc((void**)&results_d, rolloutVector.size() * sizeof(int));
	cudaMemset(boards_d, 0, rolloutVector.size() * sizeof(int));
	cudaMemcpy(boards_d, boards_to_rollout, rolloutVector.size() * sizeof(Board), cudaMemcpyHostToDevice);

	//alokalcja dynamicznych tablic w klasach
	for (int i = 0; i < rolloutVector.size(); i++)
	{
		int* hostData;
		cudaMalloc((void**)&hostData, sizeof(int) * boards_to_rollout[i].size);
		cudaMemcpy(hostData, boards_to_rollout[i].pieces, sizeof(int) * boards_to_rollout[i].size, cudaMemcpyHostToDevice);
		cudaMemcpy(&(boards_d[i].pieces), &hostData, sizeof(int*), cudaMemcpyHostToDevice);
	}

	RolloutGames<< <gridSize, blockSize >> > (boards_d, results_d, rolloutVector.size());
	cudaThreadSynchronize();
	cudaMemcpy(results, results_d, sizeof(int) * rolloutVector.size(), cudaMemcpyDeviceToHost);

	cuerr = checkCUDAError("cuda kernel");

	if (!cuerr) cuerr = checkCUDAError("cuda memcpy");

	cudaFree(boards_d);
	cudaFree(results_d);
	if (!cuerr) cuerr = checkCUDAError("cuda free");

	PRINT_ERRORS = tmp;

	return possibleMovesCount - 1;
}
