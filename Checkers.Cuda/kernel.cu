
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

std::vector<Move> GetPossibleMoves(int player, Board board)
{
	std::vector<Move> possibleMoves;
	return possibleMoves;
}

Board GetBoardAfterMove(Board board, Move move)
{
	int *pieces = new int[board.Size * board.Size];
	for (int i = 0; i != board.Size * board.Size; i++)
	{
		pieces[i] = board.Pieces[i];
	}
	for (int i = 0; i != move.BeatedPiecesCount; i++)
	{
		pieces[move.BeatedPieces[i]] = 0;
	}
	pieces[move.NewPosition] = pieces[move.OldPosition];
	pieces[move.OldPosition] = 0;
	return Board(board.Size, pieces);
}

MCTS* GenerateRoot(Board startBoard, int player, int movesCount, Move* possibleMoves)
{
	MCTS* root = new MCTS(NULL, startBoard, player);
	for (int i = 0; i != movesCount; i++)
	{
		MCTS* child = new MCTS(root, GetBoardAfterMove(startBoard, possibleMoves[i]), (player + 1) % 2);
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
		auto moves = GetPossibleMoves(leafNode->player, leafNode->board);
		if (moves.size() == 0)
			return NULL;
		for (int i = 0; i != moves.size(); i++)
		{
			leafNode->add_child(new MCTS(leafNode, GetBoardAfterMove(leafNode->board, moves[i]), (leafNode->player + 1) % 2));
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
	int player, //0 - czrny, 1 - bia造
	int* board, //0 - puste, 1 - bia造 pion, 2 - bia豉 dama, 3 - czarny pion, 4 - czarna dama
	int* possibleMoves
)
{
	Board startBoard = Board(boardSize, board);
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
	int player, //0 - bia造, 1 - czarny
	int* board, //0 - puste, 1 - bia造 pion, 2 - bia豉 dama, 3 - czarny pion, 4 - czarna dama
	int* possibleMoves
)
{
	Board startBoard = Board(boardSize, board);
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

	MCTS* root = GenerateRoot(startBoard, player, possibleMovesCount, moves);
	std::vector<MCTS*> rolloutVector;
	for (int i = 0; i != 1000; i++)
	{
		MCTS* node = SelectNode(root);
		if (node == NULL || node->visitedInCurrentIteration)
			break;
		BackpropagateSimulations(node);
		rolloutVector.push_back(node);
	}
	int tmp = PRINT_ERRORS;
	int cuerr;
	int blockSize = 1024;      // The launch configurator returned block size 
	int gridSize = 1024;       // The actual grid size needed, based on input size 

	Move *moves_d;
	Board *board_d;

	cudaMalloc((void**)&moves_d, possibleMovesCount * sizeof(Move));
	cudaMalloc((void**)&board_d, sizeof(Board));
	cudaMemcpy(moves_d, moves, possibleMovesCount * sizeof(Move), cudaMemcpyHostToDevice);
	cudaMemcpy(board_d, &startBoard, sizeof(Board), cudaMemcpyHostToDevice);

	//alokalcja dynamicznych tablic w klasach
	int* hostData;
	cudaMalloc((void**)&hostData, sizeof(int)*startBoard.Size);
	cudaMemcpy(hostData, startBoard.Pieces, sizeof(int)*startBoard.Size, cudaMemcpyHostToDevice);
	cudaMemcpy(&(board_d->Pieces), &hostData, sizeof(int *), cudaMemcpyHostToDevice);

	//alokalcja dynamicznych tablic w klasach
	for (int i = 0; i < possibleMovesCount; i++)
	{
		int* hostData;
		cudaMalloc((void**)&hostData, sizeof(int)*moves[i].BeatedPiecesCount);
		cudaMemcpy(hostData, moves[i].BeatedPieces, sizeof(int)*moves[i].BeatedPiecesCount, cudaMemcpyHostToDevice);
		cudaMemcpy(&(moves_d[i].BeatedPieces), &hostData, sizeof(int *), cudaMemcpyHostToDevice);
	}

	//cutCreateTimer(&timer);			    // from cutil.h
	//cutStartTimer(timer);
	PredictNextMove << <gridSize, blockSize >> > (board_d, moves_d);	// kernel invocation
	cudaThreadSynchronize();			// by default kernel runs in parallel with CPU code
	//cutStopTimer(timer);

	cuerr = checkCUDAError("cuda kernel");

	//cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
	if (!cuerr) cuerr = checkCUDAError("cuda memcpy");

	//sExecutionTime = cutGetTimerValue(timer);

	cudaFree(moves_d);
	if (!cuerr) cuerr = checkCUDAError("cuda free");

	PRINT_ERRORS = tmp;

	return possibleMovesCount - 1;
}
