#include "move.cuh"

__device__ __host__ Move::Move()
{

}
__device__ __host__ Move::Move(char oldPosition, char newPosition, char beatedPiecesCount, char *beatedPieces) : beated_pieces_count(beatedPiecesCount), old_position(oldPosition), new_position(newPosition)
{
	for (int i = 0; i != beatedPiecesCount; i++)
		beated_pieces[i] = beatedPieces[i];
}
__device__ __host__ Move::~Move()
{

}