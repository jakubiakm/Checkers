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

#define CUDA_CALL(ans) { GpuAssert((ans), __FILE__, __LINE__, true); }
inline void GpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ int RolloutBoard(Board board)
{
	return 1;
}

__global__ void RolloutGames(Board* rollout_boards, int* results, int size)
{
	const long numThreads = blockDim.x * gridDim.x;
	const long threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (long long ind = threadID; ind < size; ind += numThreads)
	{
		results[ind] = RolloutBoard(rollout_boards[ind]);
	}
}

__host__ Move* GetPossibleMovesFromInputParameters(int number_of_moves, int* possible_moves_array)
{
	Move* moves_to_fill = new Move[number_of_moves];
	int ind = 1;
	for (int i = 0; i != number_of_moves; i++)
	{
		int beated_pieces_count = possible_moves_array[ind++];
		int *beated_pieces = new int[beated_pieces_count];
		for (int j = 0; j != beated_pieces_count; j++)
		{
			beated_pieces[j] = possible_moves_array[ind++];
		}
		moves_to_fill[i] = Move(
			possible_moves_array[ind++],
			possible_moves_array[ind++],
			beated_pieces_count,
			beated_pieces
		);
	}
	return moves_to_fill;
}

extern "C" int __declspec(dllexport) __stdcall MakeMoveGpu
(
	int board_size,
	int current_player,			//0 - bia�y, 1 - czarny
	int* board,					//0 - puste, 1 - bia�y pion, 2 - bia�a dama, 3 - czarny pion, 4 - czarna dama
	int* possible_moves
)
{
	Player player = current_player == 0 ? Player::WHITE : Player::BLACK;		//gracz dla kt�rego wybierany jest optymalny ruch
	int
		possible_moves_count = possible_moves[0],								//liczba mo�liwych ruch�w spo�r�d kt�rych wybierany jest najlepszy
		block_size = 1024,														//rozmiar gridu z kt�rego gpu ma korzysta�
		grid_size = 1024,														//rozmiar bloku z kt�rego gpu ma korzysta� 
		*results_d,																//wska�nik na pami�� w GPU przechowuj�cy wyniki symulacji w danej iteracji
		*results;																//wska�nik na pami�� w CPU przechowuj�cy wyniki symulacji w danej iteracji
	Board
		start_board = Board(board_size, board, player),							//pocz�tkowy stan planszy
		*boards_d,																//wska�nik na pami�� w GPU przechowuj�cy plansze do symulacji
		*boards_to_rollout;														//wska�nik na pami�� w CPU przechowuj�cy plansze do symulacji
	Move *moves;																//lista mo�liwych do wykonania ruch�w
	std::vector<MctsNode*> rollout_vector;										//wektor przechowuj�cy elementy, dla kt�rych powinna zosta� wykonana symulacja dla GPU
	Mcts mcts_algorithm = Mcts();												//algorytm wybieraj�cy optymalny ruch

	moves = GetPossibleMovesFromInputParameters(possible_moves_count, possible_moves);
	mcts_algorithm.GenerateRoot(start_board, possible_moves_count, moves);

	for (int i = 0; i != block_size * grid_size; i++)
	{
		MctsNode* node = mcts_algorithm.SelectNode(mcts_algorithm.root);
		if (node == 0 || node->visited_in_current_iteration)
			break;
		mcts_algorithm.BackpropagateSimulations(node);
		rollout_vector.push_back(node);
	}

	results = new int[rollout_vector.size()];
	boards_to_rollout = new Board[rollout_vector.size()];

	for (int i = 0; i != rollout_vector.size(); i++)
	{
		boards_to_rollout[i] = rollout_vector[i]->board;
	}

	CUDA_CALL(cudaMalloc((void**)&boards_d, rollout_vector.size() * sizeof(Board)));
	CUDA_CALL(cudaMalloc((void**)&results_d, rollout_vector.size() * sizeof(int)));
	CUDA_CALL(cudaMemset(boards_d, 0, rollout_vector.size() * sizeof(int)));
	CUDA_CALL(cudaMemcpy(boards_d, boards_to_rollout, rollout_vector.size() * sizeof(Board), cudaMemcpyHostToDevice));

	//alokalcja dynamicznych tablic w klasach
	for (int i = 0; i < rollout_vector.size(); i++)
	{
		int* hostData;
		CUDA_CALL(cudaMalloc((void**)&hostData, sizeof(int) * boards_to_rollout[i].size));
		CUDA_CALL(cudaMemcpy(hostData, boards_to_rollout[i].pieces, sizeof(int) * boards_to_rollout[i].size, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(&(boards_d[i].pieces), &hostData, sizeof(int*), cudaMemcpyHostToDevice));
	}

	RolloutGames<< <grid_size, block_size >> > (boards_d, results_d, rollout_vector.size());
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaThreadSynchronize());
	CUDA_CALL(cudaMemcpy(results, results_d, sizeof(int) * rollout_vector.size(), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(boards_d));
	CUDA_CALL(cudaFree(results_d));
	CUDA_CALL(cudaDeviceReset());
	return possible_moves_count - 1;
}
