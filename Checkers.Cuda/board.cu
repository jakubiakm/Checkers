#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>

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


__device__ __host__ Move* Board::GetPossibleMovesGpu(int &moves_count)
{
	Move* moves = new Move[1];
	moves[0] = Move(0, 0, 0, 0);
	return moves;
}

__device__ __host__ Move* Board::GetPossibleMoves(int &moves_count)
{
	moves_count = 0;
	Move
		*possible_moves = new Move[1000],
		*all_possible_moves = new Move[1000],
		*kings_moves,
		*pawns_moves;
	int
		*pawn_positions = new int[100],
		*king_positions = new int[100],
		pawn_ind = 0,
		king_ind = 0,
		pawns_moves_count = 0,
		kings_moves_count = 0,
		maximal_beat_count = 0,
		temp_size = 0;
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
	for (int i = 0; i != pawn_ind; i++)
	{
		pawns_moves_count = 0;
		pawns_moves = GetPawnPossibleMoves(pawn_positions[i], pawns_moves_count);
		for (int j = 0; j != pawns_moves_count; j++)
		{
			all_possible_moves[moves_count++] = pawns_moves[j];
		}
		delete[] pawns_moves;
	}
	for (int i = 0; i != king_ind; i++)
	{
		kings_moves_count = 0;
		kings_moves = GetKingPossibleMoves(king_positions[i], kings_moves_count);
		for (int j = 0; j != kings_moves_count; j++)
		{
			all_possible_moves[moves_count++] = kings_moves[j];
		}
		delete[] kings_moves;
	}
	for (int i = 0; i != moves_count; i++)
	{
		if (all_possible_moves[i].beated_pieces_count > maximal_beat_count)
			maximal_beat_count = all_possible_moves[i].beated_pieces_count;
	}
	temp_size = moves_count;
	moves_count = 0;
	for (int i = 0; i != temp_size; i++)
	{
		if (all_possible_moves[i].beated_pieces_count == maximal_beat_count)
			possible_moves[moves_count++] = all_possible_moves[i];
		else
			delete[] all_possible_moves[i].beated_pieces;
	}

	delete[] king_positions;
	delete[] pawn_positions;
	delete[] all_possible_moves;
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
	//zamiana na damkê
	if (_pieces[move.new_position] == 1 && move.new_position >= 1 && move.new_position <= Board::size / 2)
	{
		_pieces[move.new_position] = 2;
	}
	if (_pieces[move.new_position] == 3 && move.new_position > Board::size * Board::size / 2 - Board::size / 2 && move.new_position <= Board::size * Board::size / 2)
	{
		_pieces[move.new_position] = 4;
	}
	return Board(size, _pieces, player == Player::WHITE ? Player::BLACK : Player::WHITE);
}

__device__ __host__ Move* Board::GetPawnPossibleMoves(int position, int &moves_count)
{
	Move* possible_moves = new Move[100];
	int
		piece_row = Board::PositionToRow(position),
		piece_column = Board::PositionToColumn(position);
	switch (Board::player)
	{
	case Player::WHITE:
		if (Board::CanMoveToPosition(piece_row + 1, piece_column + 1, position))
			possible_moves[moves_count++] = Move(position, Board::ToPosition(piece_row + 1, piece_column + 1), 0, 0);
		if (Board::CanMoveToPosition(piece_row + 1, piece_column - 1, position))
			possible_moves[moves_count++] = Move(position, Board::ToPosition(piece_row + 1, piece_column - 1), 0, 0);
		break;
	case Player::BLACK:
		if (Board::CanMoveToPosition(piece_row - 1, piece_column + 1, position))
			possible_moves[moves_count++] = Move(position, Board::ToPosition(piece_row - 1, piece_column + 1), 0, 0);
		if (Board::CanMoveToPosition(piece_row - 1, piece_column - 1, position))
			possible_moves[moves_count++] = Move(position, Board::ToPosition(piece_row - 1, piece_column - 1), 0, 0);
		break;
	}

	//próba bicia w czterech ró¿nych kierunkach
	if (Board::CanBeatPiece(piece_row, piece_column, piece_row - 1, piece_column - 1, position))
	{
		int* beated_pieces = new int[100];
		beated_pieces[0] = Board::ToPosition(piece_row - 1, piece_column - 1);
		Board::GetAllBeatMoves(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row - 2, piece_column - 2, possible_moves, moves_count);
	}
	if (Board::CanBeatPiece(piece_row, piece_column, piece_row + 1, piece_column - 1, position))
	{
		int* beated_pieces = new int[100];
		beated_pieces[0] = Board::ToPosition(piece_row + 1, piece_column - 1);
		Board::GetAllBeatMoves(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row + 2, piece_column - 2, possible_moves, moves_count);
	}
	if (Board::CanBeatPiece(piece_row, piece_column, piece_row - 1, piece_column + 1, position))
	{
		int* beated_pieces = new int[100];
		beated_pieces[0] = Board::ToPosition(piece_row - 1, piece_column + 1);
		Board::GetAllBeatMoves(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row - 2, piece_column + 2, possible_moves, moves_count);
	}
	if (Board::CanBeatPiece(piece_row, piece_column, piece_row + 1, piece_column + 1, position))
	{
		int* beated_pieces = new int[100];
		beated_pieces[0] = Board::ToPosition(piece_row + 1, piece_column + 1);
		Board::GetAllBeatMoves(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row + 2, piece_column + 2, possible_moves, moves_count);
	}
	return possible_moves;
}

__device__ __host__ Move* Board::GetKingPossibleMoves(int position, int &moves_count)
{
	Move* possible_moves = new Move[10000];
	int
		piece_row = Board::PositionToRow(position),
		piece_column = Board::PositionToColumn(position);

	//normalne ruchy w czterech kierunkach a¿ do napotkania pionka lub koñca planszy
	for (int i = 1; i < Board::size; i++)
	{
		if (Board::CanMoveToPosition(piece_row + i, piece_column + i, position))
			possible_moves[moves_count++] = Move(position, Board::ToPosition(piece_row + i, piece_column + i), 0, 0);
		else
			break;
	}
	for (int i = 1; i < Board::size; i++)
	{
		if (Board::CanMoveToPosition(piece_row + i, piece_column - i, position))
			possible_moves[moves_count++] = Move(position, Board::ToPosition(piece_row + i, piece_column - i), 0, 0);
		else
			break;
	}
	for (int i = 1; i < Board::size; i++)
	{
		if (Board::CanMoveToPosition(piece_row - i, piece_column + i, position))
			possible_moves[moves_count++] = Move(position, Board::ToPosition(piece_row - i, piece_column + i), 0, 0);
		else
			break;
	}
	for (int i = 1; i < Board::size; i++)
	{
		if (Board::CanMoveToPosition(piece_row - i, piece_column - i, position))
			possible_moves[moves_count++] = Move(position, Board::ToPosition(piece_row - i, piece_column - i), 0, 0);
		else
			break;
	}

	//próba bicia w czterech ró¿nych kierunkach damk¹
	for (int ind = 1; ind < Board::size; ind++)
		if (Board::CanBeatPiece(piece_row, piece_column, piece_row - ind, piece_column - ind, position))
		{
			int* beated_pieces = new int[100];
			beated_pieces[0] = Board::ToPosition(piece_row - ind, piece_column - ind);
			Board::GetAllKingBeatMoves(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row - ind - 1, piece_column - ind - 1, possible_moves, ind);
		}
		else
		{
			if (!Board::CanMoveToPosition(piece_row - ind, piece_column - ind, position))
				break;
		}
	for (int ind = 1; ind < Board::size; ind++)
		if (Board::CanBeatPiece(piece_row, piece_column, piece_row - ind, piece_column + ind, position))
		{
			int* beated_pieces = new int[100];
			beated_pieces[0] = Board::ToPosition(piece_row - ind, piece_column + ind);
			Board::GetAllKingBeatMoves(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row - ind - 1, piece_column + ind + 1, possible_moves, ind);
		}
		else
		{
			if (!Board::CanMoveToPosition(piece_row - ind, piece_column + ind, position))
				break;
		}
	for (int ind = 1; ind < Board::size; ind++)
		if (Board::CanBeatPiece(piece_row, piece_column, piece_row + ind, piece_column - ind, position))
		{
			int* beated_pieces = new int[100];
			beated_pieces[0] = Board::ToPosition(piece_row + ind, piece_column - ind);
			Board::GetAllKingBeatMoves(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row + ind + 1, piece_column - ind - 1, possible_moves, ind);
		}
		else
		{
			if (!Board::CanMoveToPosition(piece_row + ind, piece_column - ind, position))
				break;
		}
	for (int ind = 1; ind < Board::size; ind++)
		if (Board::CanBeatPiece(piece_row, piece_column, piece_row + ind, piece_column + ind, position))
		{
			int* beated_pieces = new int[100];
			beated_pieces[0] = Board::ToPosition(piece_row + ind, piece_column + ind);
			Board::GetAllKingBeatMoves(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row + ind + 1, piece_column + ind + 1, possible_moves, ind);
		}
		else
		{
			if (!Board::CanMoveToPosition(piece_row + ind, piece_column + ind, position))
				break;
		}
	return possible_moves;
}

__device__ __host__ void Board::GetAllBeatMoves(int piece_row, int piece_column, int *beated_pieces, int beated_pieces_length, int source_row, int source_column, int target_row, int target_column, Move* all_moves, int& all_moves_length)
{
	all_moves[all_moves_length++] = Move(Board::ToPosition(piece_row, piece_column), Board::ToPosition(target_row, target_column), beated_pieces_length, beated_pieces);
	if (Board::CanBeatPiece(target_row, target_column, target_row - 1, target_column - 1, Board::ToPosition(piece_row, piece_column)))
	{
		int beated_piece_position = Board::ToPosition(target_row - 1, target_column - 1);
		bool flag = true;
		for (int i = 0; i != beated_pieces_length; i++)
		{
			if (beated_pieces[i] == beated_piece_position)
			{
				flag = false;
			}
		}
		if (flag)
		{
			int *new_beated_pieces = new int[100];
			for (int i = 0; i != beated_pieces_length; i++)
			{
				new_beated_pieces[i] = beated_pieces[i];
			}
			new_beated_pieces[beated_pieces_length] = beated_piece_position;
			Board::GetAllBeatMoves(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row - 2, target_column - 2, all_moves, all_moves_length);
		}
	}
	if (Board::CanBeatPiece(target_row, target_column, target_row + 1, target_column - 1, Board::ToPosition(piece_row, piece_column)))
	{
		int beated_piece_position = Board::ToPosition(target_row + 1, target_column - 1);
		bool flag = true;
		for (int i = 0; i != beated_pieces_length; i++)
		{
			if (beated_pieces[i] == beated_piece_position)
			{
				flag = false;
			}
		}
		if (flag)
		{
			int *new_beated_pieces = new int[100];
			for (int i = 0; i != beated_pieces_length; i++)
			{
				new_beated_pieces[i] = beated_pieces[i];
			}
			new_beated_pieces[beated_pieces_length] = beated_piece_position;
			Board::GetAllBeatMoves(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row + 2, target_column - 2, all_moves, all_moves_length);
		}
	}
	if (Board::CanBeatPiece(target_row, target_column, target_row - 1, target_column + 1, Board::ToPosition(piece_row, piece_column)))
	{
		int beated_piece_position = Board::ToPosition(target_row - 1, target_column + 1);
		bool flag = true;
		for (int i = 0; i != beated_pieces_length; i++)
		{
			if (beated_pieces[i] == beated_piece_position)
			{
				flag = false;
			}
		}
		if (flag)
		{
			int *new_beated_pieces = new int[100];
			for (int i = 0; i != beated_pieces_length; i++)
			{
				new_beated_pieces[i] = beated_pieces[i];
			}
			new_beated_pieces[beated_pieces_length] = beated_piece_position;
			Board::GetAllBeatMoves(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row - 2, target_column + 2, all_moves, all_moves_length);
		}
	}
	if (Board::CanBeatPiece(target_row, target_column, target_row + 1, target_column + 1, Board::ToPosition(piece_row, piece_column)))
	{
		int beated_piece_position = Board::ToPosition(target_row + 1, target_column + 1);
		bool flag = true;
		for (int i = 0; i != beated_pieces_length; i++)
		{
			if (beated_pieces[i] == beated_piece_position)
			{
				flag = false;
			}
		}
		if (flag)
		{
			int *new_beated_pieces = new int[100];
			for (int i = 0; i != beated_pieces_length; i++)
			{
				new_beated_pieces[i] = beated_pieces[i];
			}
			new_beated_pieces[beated_pieces_length] = beated_piece_position;
			Board::GetAllBeatMoves(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row + 2, target_column + 2, all_moves, all_moves_length);
		}
	}
}

__device__ __host__ void Board::GetAllKingBeatMoves(int piece_row, int piece_column, int *beated_pieces, int beated_pieces_length, int source_row, int source_column, int target_row, int target_column, Move* all_moves, int& all_moves_length)
{
	all_moves[all_moves_length++] = Move(Board::ToPosition(piece_row, piece_column), Board::ToPosition(target_row, target_column), beated_pieces_length, beated_pieces);
	for (int ind = 1; ind < Board::size; ind++)
	{
		if (target_row - source_row > 0 && target_column - source_column > 0)
			if (Board::CanMoveToPosition(target_row + ind, target_column + ind, Board::ToPosition(piece_row, piece_column)))
			{
				int *new_beated_pieces = new int[100];
				for (int i = 0; i != beated_pieces_length; i++)
				{
					new_beated_pieces[i] = beated_pieces[i];
				}
				Board::GetAllKingBeatMoves(piece_row, piece_column, new_beated_pieces, beated_pieces_length, target_row, target_column, target_row + ind, target_column + ind, all_moves, all_moves_length);
			}
			else
				break;
		if (target_row - source_row > 0 && target_column - source_column < 0)
			if (Board::CanMoveToPosition(target_row + ind, target_column - ind, Board::ToPosition(piece_row, piece_column)))
			{
				int *new_beated_pieces = new int[100];
				for (int i = 0; i != beated_pieces_length; i++)
				{
					new_beated_pieces[i] = beated_pieces[i];
				}
				Board::GetAllKingBeatMoves(piece_row, piece_column, new_beated_pieces, beated_pieces_length, target_row, target_column, target_row + ind, target_column - ind, all_moves, all_moves_length);
			}
			else
				break;
		if (target_row - source_row < 0 && target_column - source_column > 0)
			if (Board::CanMoveToPosition(target_row - ind, target_column + ind, Board::ToPosition(piece_row, piece_column)))
			{
				int *new_beated_pieces = new int[100];
				for (int i = 0; i != beated_pieces_length; i++)
				{
					new_beated_pieces[i] = beated_pieces[i];
				}
				Board::GetAllKingBeatMoves(piece_row, piece_column, new_beated_pieces, beated_pieces_length, target_row, target_column, target_row - ind, target_column + ind, all_moves, all_moves_length);
			}
			else
				break;
		if (target_row - source_row < 0 && target_column - source_column < 0)
			if (Board::CanMoveToPosition(target_row - ind, target_column - ind, Board::ToPosition(piece_row, piece_column)))
			{
				int *new_beated_pieces = new int[100];
				for (int i = 0; i != beated_pieces_length; i++)
				{
					new_beated_pieces[i] = beated_pieces[i];
				}
				Board::GetAllKingBeatMoves(piece_row, piece_column, new_beated_pieces, beated_pieces_length, target_row, target_column, target_row - ind, target_column - ind, all_moves, all_moves_length);
			}
			else
				break;
	}
	if (!(target_row - source_row > 0 && target_column - source_column > 0))
		for (int ind = 1; ind < Board::size; ind++)
			if (Board::CanBeatPiece(target_row, target_column, target_row - ind, target_column - ind, Board::ToPosition(piece_row, piece_column)))
			{
				int beated_piece_position = Board::ToPosition(target_row - ind, target_column - ind);
				bool flag = true;
				for (int i = 0; i != beated_pieces_length; i++)
				{
					if (beated_pieces[i] == beated_piece_position)
					{
						flag = false;
					}
				}
				if (flag)
				{
					int *new_beated_pieces = new int[100];
					for (int i = 0; i != beated_pieces_length; i++)
					{
						new_beated_pieces[i] = beated_pieces[i];
					}
					new_beated_pieces[beated_pieces_length] = beated_piece_position;
					Board::GetAllKingBeatMoves(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row - ind - 1, target_column - ind - 1, all_moves, all_moves_length);
				}
				else
					break;
			}
			else
			{
				if (!Board::CanMoveToPosition(target_row - ind, target_column - ind, Board::ToPosition(piece_row, piece_column)))
					break;
			}
	if (!(target_row - source_row < 0 && target_column - source_column > 0))
		for (int ind = 1; ind < Board::size; ind++)
			if (Board::CanBeatPiece(target_row, target_column, target_row + ind, target_column - ind, Board::ToPosition(piece_row, piece_column)))
			{
				int beated_piece_position = Board::ToPosition(target_row + ind, target_column - ind);
				bool flag = true;
				for (int i = 0; i != beated_pieces_length; i++)
				{
					if (beated_pieces[i] == beated_piece_position)
					{
						flag = false;
					}
				}
				if (flag)
				{
					int *new_beated_pieces = new int[100];
					for (int i = 0; i != beated_pieces_length; i++)
					{
						new_beated_pieces[i] = beated_pieces[i];
					}
					new_beated_pieces[beated_pieces_length] = beated_piece_position;
					Board::GetAllKingBeatMoves(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row + ind + 1, target_column - ind - 1, all_moves, all_moves_length);
				}
				else
					break;
			}
			else
			{
				if (!Board::CanMoveToPosition(target_row + ind, target_column - ind, Board::ToPosition(piece_row, piece_column)))
					break;
			}
	if (!(target_row - source_row > 0 && target_column - source_column < 0))
		for (int ind = 1; ind < Board::size; ind++)
			if (Board::CanBeatPiece(target_row, target_column, target_row - ind, target_column + ind, Board::ToPosition(piece_row, piece_column)))
			{
				int beated_piece_position = Board::ToPosition(target_row - ind, target_column + ind);
				bool flag = true;
				for (int i = 0; i != beated_pieces_length; i++)
				{
					if (beated_pieces[i] == beated_piece_position)
					{
						flag = false;
					}
				}
				if (flag)
				{
					int *new_beated_pieces = new int[100];
					for (int i = 0; i != beated_pieces_length; i++)
					{
						new_beated_pieces[i] = beated_pieces[i];
					}
					new_beated_pieces[beated_pieces_length] = beated_piece_position;
					Board::GetAllKingBeatMoves(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row - ind - 1, target_column + ind + 1, all_moves, all_moves_length);
				}
				else
					break;
			}
			else
			{
				if (!Board::CanMoveToPosition(target_row - ind, target_column + ind, Board::ToPosition(piece_row, piece_column)))
					break;
			}
	if (!(target_row - source_row < 0 && target_column - source_column < 0))
		for (int ind = 1; ind < Board::size; ind++)
			if (CanBeatPiece(target_row, target_column, target_row + ind, target_column + ind, Board::ToPosition(piece_row, piece_column)))
			{
				int beated_piece_position = Board::ToPosition(target_row + ind, target_column + ind);
				bool flag = true;
				for (int i = 0; i != beated_pieces_length; i++)
				{
					if (beated_pieces[i] == beated_piece_position)
					{
						flag = false;
					}
				}
				if (flag)
				{
					int *new_beated_pieces = new int[100];
					for (int i = 0; i != beated_pieces_length; i++)
					{
						new_beated_pieces[i] = beated_pieces[i];
					}
					new_beated_pieces[beated_pieces_length] = beated_piece_position;
					Board::GetAllKingBeatMoves(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row + ind + 1, target_column + ind + 1, all_moves, all_moves_length);
				}
				else
					break;
			}
			else
			{
				if (!Board::CanMoveToPosition(target_row + ind, target_column + ind, Board::ToPosition(piece_row, piece_column)))
					break;
			}
}

__device__ __host__ bool Board::CanMoveToPosition(int position_row, int position_column, int source_move_position)
{
	int position = Board::ToPosition(position_row, position_column);
	return
		position_row >= 0 &&
		position_column >= 0 &&
		position_row <= Board::size - 1 &&
		position_column <= Board::size - 1 &&
		(
			Board::pieces[position] == 0 ||
			position == source_move_position //bij¹cy pionek nie powinien byæ brany pod uwagê
			);
}

__device__ __host__ bool Board::CanBeatPiece(int position_row, int position_column, int target_piece_position_row, int target_piece_position_column, int source_move_position)
{
	int
		target_piece_position = ToPosition(target_piece_position_row, target_piece_position_column),
		row_after_beat = target_piece_position_row + (target_piece_position_row - position_row > 0 ? 1 : -1),
		column_after_beat = target_piece_position_column + (target_piece_position_column - position_column > 0 ? 1 : -1);

	//sprawdzenie czy bite pole i pole po biciu mieszcz¹ siê w planszy
	if (!(
		target_piece_position_row >= 0 &&
		target_piece_position_row < Board::size &&
		target_piece_position_column >= 0 &&
		target_piece_position_column < Board::size &&
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
			) && CanMoveToPosition(row_after_beat, column_after_beat, source_move_position);
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

__device__ __host__ int Board::ToPosition(int row, int column)
{
	if ((column % 2 == 0 && row % 2 == 1) || (row % 2 == 0 && column % 2 == 1))
		return -1;
	else
		return size / 2 * (size - row - 1) + ((row % 2 == 0) ? 1 : 0) + (column + 1) / 2;
}

__device__ __host__ Player Board::Rollout()
{
	int
		moves_count = 0,
		move_ind = 0;
	Board current_board = *this;
	while (1)
	{
		if (current_board.IsGameFinished())
			break;
		Move *possible_moves = current_board.GetPossibleMoves(moves_count);
		if (moves_count == 0)
			break;
		move_ind = rand() % moves_count;
		Board new_board = current_board.GetBoardAfterMove(possible_moves[move_ind]);
		delete[] current_board.pieces;
		current_board = new_board;
		for (int i = 0; i != moves_count; i++)
		{
			if (possible_moves[i].beated_pieces_count > 0)
				delete[] possible_moves[i].beated_pieces;
		}
		delete[] possible_moves;
	}
	for (int i = 0; i != Board::size * Board::size; i++)
	{
		if (current_board.pieces[i] == 1 || current_board.pieces[i] == 2)
			return Player::WHITE;
		if (current_board.pieces[i] == 3 || current_board.pieces[i] == 4)
			return Player::BLACK;
	}
	return Player::BLACK;
}

__device__ __host__ bool Board::IsGameFinished()
{
	int player_pieces = 0;
	for (int i = 0; i != Board::size * Board::size; i++)
	{
		if (pieces[i] == 1 || pieces[i] == 2)
		{
			if (player_pieces == 2)
				return false;
			player_pieces = 1;
		}
		if (pieces[i] == 3 || pieces[i] == 4)
		{
			if (player_pieces == 1)
				return false;
			player_pieces = 2;
		}
	}
	return true;
}