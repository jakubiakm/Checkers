#include "cuda_runtime.h"
#include "device_launch_parameters.h"


enum Player
{
	White, Black
};
class Board
{
public:
	int size;
	int* pieces;
	Player player;
	__device__ __host__ Board(int size, int* _pieces, Player player) : size(size), player(player)
	{
		pieces = new int[size * size];
		for (int i = 0; i != size * size; i++)
		{
			pieces[i] = _pieces[i];
		}
	}
	__device__ __host__ Board()
	{

	}

	__device__ __host__ ~Board()
	{

	}

	__device__ __host__ Move* Board::get_possible_moves(int &moves_count)
	{
		moves_count = 0;
		Move* possibleMoves = new Move[100];
		return possibleMoves;
	}

	__device__ __host__ Board Board::get_board_after_move(Move move)
	{
		int *_pieces = new int[size * size];
		for (int i = 0; i != size * size; i++)
		{
			_pieces[i] = pieces[i];
		}
		for (int i = 0; i != move.beatedPiecesCount; i++)
		{
			_pieces[move.beatedPieces[i]] = 0;
		}
		_pieces[move.newPosition] = _pieces[move.oldPosition];
		_pieces[move.oldPosition] = 0;
		return Board(size, _pieces, player == Player::White ? Player::Black : Player::White);
	}

};