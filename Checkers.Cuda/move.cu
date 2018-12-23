#include "move.cuh"

__device__ __host__ Move::Move()
{

}
__device__ __host__ Move::Move(int oldPosition, int newPosition, int beatedPiecesCount, int *beatedPieces) : beated_pieces_count(beatedPiecesCount), beated_pieces(beatedPieces), old_position(oldPosition), new_position(newPosition)
{

}
__device__ __host__ Move::~Move()
{

}