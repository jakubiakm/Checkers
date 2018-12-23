#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Move
{
public:
	int beated_pieces_count;
	int *beated_pieces;
	int old_position;
	int new_position;

	__device__ __host__ Move();
	__device__ __host__ Move(int oldPosition, int newPosition, int beatedPiecesCount, int *beatedPieces);
	__device__ __host__ ~Move();
};