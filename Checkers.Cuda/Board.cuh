#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "move.cuh"

enum Player
{
	WHITE = 0, 
	BLACK = 1
};

class Board
{
public:
	int size;
	int* pieces;
	Player player;

	__device__ __host__ Board(int size, int* _pieces, Player player);
	__device__ __host__ Board();
	__device__ __host__ ~Board();
	__device__ __host__ Move* GetPossibleMoves(int &moves_count);
	__device__ __host__ Board GetBoardAfterMove(Move move);
	
private:
	__device__ __host__ Move* GetPawnPossibleMoves(int *positions, int length, int &moves_count);
	__device__ __host__ Move* GetKingPossibleMoves(int *positions, int length, int &moves_count);
	__device__ __host__ Move* GetAllBeatMoves();
	__device__ __host__ bool CanMoveToPosition(int position, int source_move_position);
	__device__ __host__ bool CanBeatPiece(int position, int target_piece_position, int source_move_position);
	__device__ __host__ int PositionToRow(int position);
	__device__ __host__ int PositionToColumn(int position);
};