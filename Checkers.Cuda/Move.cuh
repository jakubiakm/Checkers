#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Move
{
public:
	int beatedPiecesCount;
	int *beatedPieces;
	int oldPosition;
	int newPosition;
	__device__ __host__ Move()
	{

	}
	__device__ __host__ Move(int oldPosition, int newPosition, int beatedPiecesCount, int *beatedPieces) : beatedPiecesCount(beatedPiecesCount), beatedPieces(beatedPieces), oldPosition(oldPosition), newPosition(newPosition)
	{

	}
	__device__ __host__ ~Move()
	{

	}
};