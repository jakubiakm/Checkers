#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Move
{
public:
	char beated_pieces_count;
	char* beated_pieces;
	char old_position;
	char new_position;

	__device__ __host__ Move();
	__device__ __host__ Move(char oldPosition, char newPosition, char beatedPiecesCount, char *beatedPieces);
	__device__ __host__ ~Move();
};