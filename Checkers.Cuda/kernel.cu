#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h> 
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "move.cuh"
#include "board.cuh"
#include "mctsnode.h"
#include "mcts.h"

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

extern "C" int __declspec(dllexport) __stdcall MakeMoveGpu
(
	int boardSize,
	int _player, //0 - bia³y, 1 - czarny
	int* board, //0 - puste, 1 - bia³y pion, 2 - bia³a dama, 3 - czarny pion, 4 - czarna dama
	int* possibleMoves
)
{
	Mcts mcts_algorithm = Mcts();
	Player player = _player == 0 ? Player::WHITE : Player::BLACK;
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

	MctsNode* root = mcts_algorithm.GenerateRoot(startBoard, possibleMovesCount, moves);
	std::vector<MctsNode*> rolloutVector;

	int tmp = PRINT_ERRORS;
	int cuerr;
	int blockSize = 1024;      // The launch configurator returned block size 
	int gridSize = 1024;       // The actual grid size needed, based on input size 

	for (int i = 0; i != blockSize * gridSize; i++)
	{
		MctsNode* node = mcts_algorithm.SelectNode(root);
		if (node == NULL || node->visited_in_current_iteration)
			break;
		mcts_algorithm.BackpropagateSimulations(node);
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
