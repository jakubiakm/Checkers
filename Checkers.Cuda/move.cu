#include "move.cuh"

__device__ __host__ Move::Move()
{

}
__device__ __host__ Move::Move(char oldPosition, char newPosition, char beatedPiecesCount, char *beatedPieces) : beated_pieces_count(beatedPiecesCount), old_position(oldPosition), new_position(newPosition), beated_pieces(beatedPieces)
{
}
__device__ __host__ Move::~Move()
{

}