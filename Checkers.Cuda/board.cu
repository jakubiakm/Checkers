#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Board.cuh"

__device__ __host__ Board::Board(int size, int* _pieces, Player player) : size(size), player(player)
{
	pieces = new int[size * size];
	for (int i = 0; i != size * size; i++)
	{
		pieces[i] = _pieces[i];
	}
}

__device__ __host__ Board::Board()
{

}

__device__ __host__ Board::~Board()
{

}

__device__ __host__ Move* Board::GetPossibleMoves(int &moves_count)
{
	moves_count = 0;
	Move* possibleMoves = new Move[100];
	return possibleMoves;
}

__device__ __host__ Board Board::GetBoardAfterMove(Move move)
{
	int *_pieces = new int[size * size];
	for (int i = 0; i != size * size; i++)
	{
		_pieces[i] = pieces[i];
	}
	for (int i = 0; i != move.beated_pieces_count; i++)
	{
		_pieces[move.beated_pieces[i]] = 0;
	}
	_pieces[move.new_position] = _pieces[move.old_position];
	_pieces[move.old_position] = 0;
	return Board(size, _pieces, player == Player::WHITE ? Player::BLACK : Player::WHITE);
}