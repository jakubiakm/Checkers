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
	Move
		*possible_moves,
		*kings_moves,
		*pawns_moves;
	int
		*pawn_positions = new int[100],
		*king_positions = new int[100],
		pawn_ind = 0,
		king_ind = 0,
		pawns_moves_count = 0,
		kings_moves_count = 0;
	for (int i = 0; i != Board::size * Board::size; i++)
	{
		if (Board::player == Player::BLACK)
		{
			if (Board::pieces[i] == 3)
			{
				pawn_positions[pawn_ind++] = i;
			}
			if (Board::pieces[i] == 4)
			{
				king_positions[king_ind++] = i;
			}
		}
		else
		{
			if (Board::pieces[i] == 1)
			{
				pawn_positions[pawn_ind++] = i;
			}
			if (Board::pieces[i] == 2)
			{
				king_positions[king_ind++] = i;
			}
		}
	}
	pawns_moves = GetPawnPossibleMoves(pawn_positions, pawn_ind, pawns_moves_count);
	kings_moves = GetKingPossibleMoves(king_positions, king_ind, kings_moves_count);
	possible_moves = new Move[pawns_moves_count + kings_moves_count];
	for (int i = 0; i != pawns_moves_count; i++)
	{
		possible_moves[moves_count++] = pawns_moves[i++];
	}
	for (int i = 0; i != kings_moves_count; i++)
	{
		possible_moves[moves_count++] = kings_moves[i++];
	}
	return possible_moves;
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

__device__ __host__ Move* Board::GetPawnPossibleMoves(int *positions, int length, int &moves_count)
{
	Move* possibleMoves = new Move[100];

	return possibleMoves;
}

__device__ __host__ Move* Board::GetKingPossibleMoves(int *positions, int length, int &moves_count)
{
	Move* possibleMoves = new Move[100];

	return possibleMoves;
}

__device__ __host__ Move* Board::GetAllBeatMoves()
{
	Move* possibleMoves = new Move[100];

	return possibleMoves;
}

__device__ __host__ bool Board::CanMoveToPosition(int position, int source_move_position)
{
	return
		position >= 1 &&
		position <= (Board::size * Board::size) / 2 &&
		(
			Board::pieces[position] == 0 ||
			position == source_move_position //bij¹cy pionek nie powinien byæ brany pod uwagê
			);
}

__device__ __host__ bool Board::CanBeatPiece(int position, int target_piece_position, int source_move_position)
{
	int
		row = Board::PositionToRow(position),
		column = Board::PositionToColumn(position),
		target_row = Board::PositionToRow(target_piece_position),
		target_column = Board::PositionToColumn(target_piece_position),
		row_after_beat = target_row + (target_row - row > 0 ? 1 : -1),
		column_after_beat = target_column + (target_column - column > 0 ? 1 : -1),
		position_after_beat = size / 2 * (size - row - 1) + ((row % 2 == 0) ? 1 : 0) + (column + 1) / 2;

	//sprawdzenie czy bite pole i pole po biciu mieszcz¹ siê w planszy
	if (!(
		target_row >= 0 &&
		target_row < Board::size &&
		target_column >= 0 &&
		target_column < Board::size &&
		row_after_beat >= 0 &&
		row_after_beat < Board::size &&
		column_after_beat >= 0 &&
		column_after_beat < Board::size))
		return false;

	//sprawdzenie czy jest przeciwny pionek na pozycji i czy po biciu mo¿na postawiæ pionka na nastêpnym polu
	return
		(
			target_piece_position != source_move_position &&
			(
				Board::player == Player::BLACK &&
				(
					Board::pieces[target_piece_position] == 1 ||
					Board::pieces[target_piece_position] == 2) 
				||
				Board::player == Player::WHITE &&
				(
					Board::pieces[target_piece_position] == 3 ||
					Board::pieces[target_piece_position] == 4)
				)
			) && CanMoveToPosition(position_after_beat, source_move_position);
}

__device__ __host__ int Board::PositionToColumn(int position)
{
	int div = size / 2;
	return 2 * ((position - 1) % div) + (Board::PositionToRow(position) % 2 == 1 ? 1 : 0);
}

__device__ __host__ int Board::PositionToRow(int position)
{
	int div = Board::size / 2;
	return Board::size - 1 - (position - 1) / div;
}