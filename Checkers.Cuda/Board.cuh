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
	__device__ __host__ Move* Board::GetPossibleMoves(int &moves_count);
	__device__ __host__ Board Board::GetBoardAfterMove(Move move);
};