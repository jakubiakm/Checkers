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

__global__ void RolloutGames(Board* rollout_boards, int* results, int size)
{
	const long numThreads = blockDim.x * gridDim.x;
	const long threadID = blockIdx.x * blockDim.x + threadIdx.x;


	for (long long ind = threadID; ind < size; ind += numThreads)
	{
		Board current_board = rollout_boards[ind];

		Player player = current_board.Rollout();
		results[ind] = player == Player::BLACK ? 1 : 0;
	}
}

__host__ Move* GetPossibleMovesFromInputParameters(int number_of_moves, char* possible_moves_array)
{
	Move *moves_to_fill = new Move[100];
	int ind = 1;
	for (int i = 0; i != number_of_moves; i++)
	{
		char beated_pieces_count = possible_moves_array[ind++];
		char *beated_pieces = new char[10];
		for (int j = 0; j != beated_pieces_count; j++)
		{
			beated_pieces[j] = possible_moves_array[ind++];
		}
		char old_position = possible_moves_array[ind++];
		char new_position = possible_moves_array[ind++];
		moves_to_fill[i] = Move(
			old_position,
			new_position,
			beated_pieces_count,
			beated_pieces
		);
	}
	return moves_to_fill;
}

__host__ void DeallocateMctsNode(MctsNode *node)
{
	for (int i = 0; i != node->children.size(); i++)
	{
		DeallocateMctsNode(node->children[i]);
	}
	//delete [] node->board.pieces;
	delete node;
}

extern "C" int __declspec(dllexport) __stdcall MakeMoveGpu
(
	char board_size,
	int current_player,			//0 - bia³y, 1 - czarny
	char* board,					//0 - puste, 1 - bia³y pion, 2 - bia³a dama, 3 - czarny pion, 4 - czarna dama
	char* possible_moves
)
{
	Player player = current_player == 0 ? Player::WHITE : Player::BLACK;	//gracz dla którego wybierany jest optymalny ruch
	int
		number_of_mcts_iterations = 25,										//liczba iteracji wykonana przez algorytm MCTS
		possible_moves_count = possible_moves[0],							//liczba mo¿liwych ruchów spoœród których wybierany jest najlepszy
		block_size = 1024,													//rozmiar gridu z którego gpu ma korzystaæ
		grid_size = 1024,													//rozmiar bloku z którego gpu ma korzystaæ 
		*results_d,															//wskaŸnik na pamiêæ w GPU przechowuj¹cy wyniki symulacji w danej iteracji
		*results;															//wskaŸnik na pamiêæ w CPU przechowuj¹cy wyniki symulacji w danej iteracji
	Board
		start_board = Board(board_size, board, player),						//pocz¹tkowy stan planszy
		*boards_d,															//wskaŸnik na pamiêæ w GPU przechowuj¹cy plansze do symulacji
		*boards_to_rollout;													//wskaŸnik na pamiêæ w CPU przechowuj¹cy plansze do symulacji
	Move* moves;															//lista mo¿liwych do wykonania ruchów
	std::vector<MctsNode*> rollout_vector;									//wektor przechowuj¹cy elementy, dla których powinna zostaæ wykonana symulacja dla GPU
	Mcts mcts_algorithm = Mcts();											//algorytm wybieraj¹cy optymalny ruch

	moves = GetPossibleMovesFromInputParameters(possible_moves_count, possible_moves);

	mcts_algorithm.GenerateRoot(start_board, possible_moves_count, moves);
	for (int i = 0; i != possible_moves_count; i++)
		if (moves[i].beated_pieces_count > 0)
			delete[] moves[i].beated_pieces;
	delete[] moves;

	while (number_of_mcts_iterations--)
	{
		rollout_vector.clear();
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

		size_t size;
		CUDA_CALL(cudaDeviceGetLimit(&size, cudaLimitStackSize));
		CUDA_CALL(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
		RolloutGames << <4, 256>> > (boards_d, results_d, rollout_vector.size());
		
		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaGetLastError());
		CUDA_CALL(cudaMemcpy(results, results_d, sizeof(int) * rollout_vector.size(), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaFree(boards_d));
		CUDA_CALL(cudaFree(results_d));
		mcts_algorithm.BackpropagateResults(rollout_vector, results);

		delete[] results;
		delete[] boards_to_rollout;
	}

	int best_move = mcts_algorithm.GetBestMove();
	DeallocateMctsNode(mcts_algorithm.root);
	return best_move;
}

