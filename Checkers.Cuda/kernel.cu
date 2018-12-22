
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h> 
#include <vector>
//#include <cutil.h>		// timers
#include "board.h"
#include "Move.h"
#include "MCTS.h"

#define cConst 2

int N = 0;

Board GetBoardAfterMove(Board board, Move move)
{
	int *pieces = new int[board.size * board.size];
	for (int i = 0; i != board.size * board.size; i++)
	{
		pieces[i] = board.pieces[i];
	}
	for (int i = 0; i != move.beatedPiecesCount; i++)
	{
		pieces[move.beatedPieces[i]] = 0;
	}
	pieces[move.newPosition] = pieces[move.oldPosition];
	pieces[move.oldPosition] = 0;
	return Board(board.size, pieces, board.player == Player::White ? Player::Black : Player::White);
}

MCTS* GenerateRoot(Board startBoard, int movesCount, Move* possibleMoves)
{
	MCTS* root = new MCTS(NULL, startBoard);
	for (int i = 0; i != movesCount; i++)
	{
		MCTS* child = new MCTS(root, GetBoardAfterMove(startBoard, possibleMoves[i]));
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
		if (leafNode->visitedInCurrentIteration)
			return NULL;
		auto moves = leafNode->board.get_possible_moves();
		if (moves.size() == 0)
			return NULL;
		for (int i = 0; i != moves.size(); i++)
		{
			leafNode->add_child(new MCTS(leafNode, GetBoardAfterMove(leafNode->board, moves[i])));
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

__global__ void PredictNextMove(Board *board, Move* startingMoves)
{

}

// cuda kernel (internal)
__global__ void some_calculations(float *a, unsigned int N, unsigned int M)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		// note1: no need for shared memory here
		// note2: global memory access is coalesced
		//        (no structs, float only used)

		// do computations M times on each thread
		// to extend processor time
		for (unsigned int i = 0; i < M; i++)
		{
			// some easy arithmetics		
			a[idx] = a[idx] * a[idx] * 0.1 - a[idx] - 10;
		}
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

	Move *moves_d;
	Board *board_d;

	cudaMalloc((void**)&moves_d, possibleMovesCount * sizeof(Move));
	cudaMalloc((void**)&board_d, sizeof(Board));
	cudaMemcpy(moves_d, moves, possibleMovesCount * sizeof(Move), cudaMemcpyHostToDevice);
	cudaMemcpy(board_d, &startBoard, sizeof(Board), cudaMemcpyHostToDevice);

	//alokalcja dynamicznych tablic w klasach
	int* hostData;
	cudaMalloc((void**)&hostData, sizeof(int)*startBoard.size);
	cudaMemcpy(hostData, startBoard.pieces, sizeof(int)*startBoard.size, cudaMemcpyHostToDevice);
	cudaMemcpy(&(board_d->pieces), &hostData, sizeof(int *), cudaMemcpyHostToDevice);

	//alokalcja dynamicznych tablic w klasach
	for (int i = 0; i < possibleMovesCount; i++)
	{
		int* hostData;
		cudaMalloc((void**)&hostData, sizeof(int)*moves[i].beatedPiecesCount);
		cudaMemcpy(hostData, moves[i].beatedPieces, sizeof(int)*moves[i].beatedPiecesCount, cudaMemcpyHostToDevice);
		cudaMemcpy(&(moves_d[i].beatedPieces), &hostData, sizeof(int *), cudaMemcpyHostToDevice);
	}

	PredictNextMove << <gridSize, blockSize >> > (board_d, moves_d);	// kernel invocation
	cudaThreadSynchronize();			// by default kernel runs in parallel with CPU code
	
	cuerr = checkCUDAError("cuda kernel");

	if (!cuerr) cuerr = checkCUDAError("cuda memcpy");

	cudaFree(moves_d);
	if (!cuerr) cuerr = checkCUDAError("cuda free");

	PRINT_ERRORS = tmp;

	return possibleMovesCount - 1;
}
