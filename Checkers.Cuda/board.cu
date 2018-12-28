#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

#include "Board.cuh"

__device__ __host__ Board::Board(char size, char* _pieces, Player player) : size(size), player(player)
{
	for (int i = 0; i != 100; i++)
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

__host__ Move* Board::GetPossibleMovesCpu(int &moves_count)
{
	moves_count = 0;
	Move
		*possible_moves = new Move[70],
		*all_possible_moves = new Move[200],
		*kings_moves,
		*pawns_moves;
	int
		pawn_ind = 0,
		king_ind = 0,
		pawns_moves_count = 0,
		kings_moves_count = 0,
		maximal_beat_count = 0,
		temp_size = 0;
	char
		*pawn_positions = new char[25],
		*king_positions = new char[25];
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
		pawns_moves = GetPawnPossibleMovesCpu(pawn_positions[i], pawns_moves_count);
		for (int j = 0; j != pawns_moves_count; j++)
		{
			all_possible_moves[moves_count++] = pawns_moves[j];
		}
		delete[] pawns_moves;
	}

	for (int i = 0; i != king_ind; i++)
	{
		kings_moves_count = 0;
		kings_moves = GetKingPossibleMovesCpu(king_positions[i], kings_moves_count);
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

	}
	delete[] all_possible_moves;
	delete[] pawn_positions;
	delete[] king_positions;

	return possible_moves;
}

__device__ void Board::GetPossibleMovesGpu(int &moves_count, Move *all_moves_device, int thread_id)
{
	/*
	dla ka�dego threada alokowane jest 1000 element�w Move. W ostatnich 200 elementach z tych 1000
	przechowywane s� tymczasowe ruchy dla pion�w i dam. Po wykonaniu funkcji elementy od
	[thread_id * 1000] do [thread_id * 1000 + moves_count] zawieraj� dozwolone ruchy
	*/
	moves_count = 0;
	int
		pawns_counter = 0,
		kings_counter = 0,
		pawn_ind = 0,
		king_ind = 0,
		pawns_moves_count = 0,
		kings_moves_count = 0,
		maximal_beat_count = 0;
	char
		pawn_positions[25],
		king_positions[25];
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
		GetPawnPossibleMovesGpu(pawn_positions[i], pawns_moves_count, all_moves_device, thread_id);
		for (int j = 0; j != pawns_moves_count; j++)
		{
			if (pawns_counter >= 100)
				all_moves_device[-1] = Move(); //wyrzuci wyj�tek jak przekroczy zamierzony rozmiar tablicy
			all_moves_device[1000 * thread_id + pawns_counter++] = all_moves_device[1000 * (thread_id + 1) - 1 - j];
		}
	}

	for (int i = 0; i != king_ind; i++)
	{
		kings_moves_count = 0;
		GetKingPossibleMovesGpu(king_positions[i], kings_moves_count, all_moves_device, thread_id);
		for (int j = 0; j != kings_moves_count; j++)
		{
			if (kings_counter >= 400)
				all_moves_device[-1] = Move(); //wyrzuci wyj�tek jak przekroczy zamierzony rozmiar tablicy
			all_moves_device[1000 * thread_id + 100 + kings_counter++] = all_moves_device[1000 * (thread_id + 1) - 1 - j];
		}
	}
	for (int i = 0; i != pawns_counter; i++)
	{
		if (all_moves_device[1000 * thread_id + i].beated_pieces_count > maximal_beat_count)
			maximal_beat_count = all_moves_device[1000 * thread_id + i].beated_pieces_count;
	}
	for (int i = 0; i != kings_counter; i++)
	{
		if (all_moves_device[1000 * thread_id + 100 + i].beated_pieces_count > maximal_beat_count)
			maximal_beat_count = all_moves_device[1000 * thread_id + 100 + i].beated_pieces_count;
	}
	moves_count = 0;
	for (int i = 0; i != pawns_counter; i++)
	{
		if (all_moves_device[1000 * thread_id + i].beated_pieces_count == maximal_beat_count)
			all_moves_device[1000 * thread_id + moves_count++] = all_moves_device[1000 * thread_id + i];
	}
	for (int i = 0; i != kings_counter; i++)
	{
		if (all_moves_device[1000 * thread_id + 100 + i].beated_pieces_count == maximal_beat_count)
			all_moves_device[1000 * thread_id + moves_count++] = all_moves_device[1000 * thread_id + 100 + i];
	}
}

__device__ __host__ Board Board::GetBoardAfterMove(Move move)
{
	char _pieces[100];
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
	//zamiana na damk�
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

__host__ Move* Board::GetPawnPossibleMovesCpu(char position, int &moves_count)
{
	Move* possible_moves = new Move[100];
	char
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

	//pr�ba bicia w czterech r�nych kierunkach
	if (Board::CanBeatPiece(piece_row, piece_column, piece_row - 1, piece_column - 1, position))
	{
		char beated_pieces[1];
		beated_pieces[0] = Board::ToPosition(piece_row - 1, piece_column - 1);
		Board::GetAllBeatMovesCpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row - 2, piece_column - 2, possible_moves, moves_count);
	}
	if (Board::CanBeatPiece(piece_row, piece_column, piece_row + 1, piece_column - 1, position))
	{
		char beated_pieces[1];
		beated_pieces[0] = Board::ToPosition(piece_row + 1, piece_column - 1);
		Board::GetAllBeatMovesCpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row + 2, piece_column - 2, possible_moves, moves_count);
	}
	if (Board::CanBeatPiece(piece_row, piece_column, piece_row - 1, piece_column + 1, position))
	{
		char beated_pieces[1];
		beated_pieces[0] = Board::ToPosition(piece_row - 1, piece_column + 1);
		Board::GetAllBeatMovesCpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row - 2, piece_column + 2, possible_moves, moves_count);
	}
	if (Board::CanBeatPiece(piece_row, piece_column, piece_row + 1, piece_column + 1, position))
	{
		char beated_pieces[1];
		beated_pieces[0] = Board::ToPosition(piece_row + 1, piece_column + 1);
		Board::GetAllBeatMovesCpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row + 2, piece_column + 2, possible_moves, moves_count);
	}
	return possible_moves;
}

__host__ Move* Board::GetKingPossibleMovesCpu(char position, int &moves_count)
{
	Move *possible_moves = new Move[200];
	char
		piece_row = Board::PositionToRow(position),
		piece_column = Board::PositionToColumn(position);

	//normalne ruchy w czterech kierunkach a� do napotkania pionka lub ko�ca planszy
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

	//pr�ba bicia w czterech r�nych kierunkach damk�
	for (int ind = 1; ind < Board::size; ind++)
		if (Board::CanBeatPiece(piece_row, piece_column, piece_row - ind, piece_column - ind, position))
		{
			char beated_pieces[1];
			beated_pieces[0] = Board::ToPosition(piece_row - ind, piece_column - ind);
			Board::GetAllKingBeatMovesCpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row - ind - 1, piece_column - ind - 1, possible_moves, moves_count);
		}
		else
		{
			if (!Board::CanMoveToPosition(piece_row - ind, piece_column - ind, position))
				break;
		}
	for (int ind = 1; ind < Board::size; ind++)
		if (Board::CanBeatPiece(piece_row, piece_column, piece_row - ind, piece_column + ind, position))
		{
			char beated_pieces[1];
			beated_pieces[0] = Board::ToPosition(piece_row - ind, piece_column + ind);
			Board::GetAllKingBeatMovesCpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row - ind - 1, piece_column + ind + 1, possible_moves, moves_count);
		}
		else
		{
			if (!Board::CanMoveToPosition(piece_row - ind, piece_column + ind, position))
				break;
		}
	for (int ind = 1; ind < Board::size; ind++)
		if (Board::CanBeatPiece(piece_row, piece_column, piece_row + ind, piece_column - ind, position))
		{
			char beated_pieces[1];
			beated_pieces[0] = Board::ToPosition(piece_row + ind, piece_column - ind);
			Board::GetAllKingBeatMovesCpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row + ind + 1, piece_column - ind - 1, possible_moves, moves_count);
		}
		else
		{
			if (!Board::CanMoveToPosition(piece_row + ind, piece_column - ind, position))
				break;
		}
	for (int ind = 1; ind < Board::size; ind++)
		if (Board::CanBeatPiece(piece_row, piece_column, piece_row + ind, piece_column + ind, position))
		{
			char beated_pieces[1];
			beated_pieces[0] = Board::ToPosition(piece_row + ind, piece_column + ind);
			Board::GetAllKingBeatMovesCpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row + ind + 1, piece_column + ind + 1, possible_moves, moves_count);
		}
		else
		{
			if (!Board::CanMoveToPosition(piece_row + ind, piece_column + ind, position))
				break;
		}
	return possible_moves;
}

__host__ void Board::GetAllBeatMovesCpu(char piece_row, char piece_column, char *beated_pieces, char beated_pieces_length, char source_row, char source_column, char target_row, char target_column, Move* all_moves, int& all_moves_length)
{
	all_moves[all_moves_length++] = Move(Board::ToPosition(piece_row, piece_column), Board::ToPosition(target_row, target_column), beated_pieces_length, beated_pieces);
	if (Board::CanBeatPiece(target_row, target_column, target_row - 1, target_column - 1, Board::ToPosition(piece_row, piece_column)))
	{
		char beated_piece_position = Board::ToPosition(target_row - 1, target_column - 1);
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
			char new_beated_pieces[10];
			for (int i = 0; i != beated_pieces_length; i++)
			{
				new_beated_pieces[i] = beated_pieces[i];
			}
			new_beated_pieces[beated_pieces_length] = beated_piece_position;
			Board::GetAllBeatMovesCpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row - 2, target_column - 2, all_moves, all_moves_length);
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
			char new_beated_pieces[10];
			for (int i = 0; i != beated_pieces_length; i++)
			{
				new_beated_pieces[i] = beated_pieces[i];
			}
			new_beated_pieces[beated_pieces_length] = beated_piece_position;
			Board::GetAllBeatMovesCpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row + 2, target_column - 2, all_moves, all_moves_length);
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
			char new_beated_pieces[10];
			for (int i = 0; i != beated_pieces_length; i++)
			{
				new_beated_pieces[i] = beated_pieces[i];
			}
			new_beated_pieces[beated_pieces_length] = beated_piece_position;
			Board::GetAllBeatMovesCpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row - 2, target_column + 2, all_moves, all_moves_length);
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
			char new_beated_pieces[10];
			for (int i = 0; i != beated_pieces_length; i++)
			{
				new_beated_pieces[i] = beated_pieces[i];
			}
			new_beated_pieces[beated_pieces_length] = beated_piece_position;
			Board::GetAllBeatMovesCpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row + 2, target_column + 2, all_moves, all_moves_length);
		}
	}
}

__host__ void Board::GetAllKingBeatMovesCpu(char piece_row, char piece_column, char *beated_pieces, char beated_pieces_length, char source_row, char source_column, char target_row, char target_column, Move* all_moves, int& all_moves_length)
{
	all_moves[all_moves_length++] = Move(Board::ToPosition(piece_row, piece_column), Board::ToPosition(target_row, target_column), beated_pieces_length, beated_pieces);
	for (int ind = 1; ind < Board::size; ind++)
	{
		if (target_row - source_row > 0 && target_column - source_column > 0)
			if (Board::CanMoveToPosition(target_row + ind, target_column + ind, Board::ToPosition(piece_row, piece_column)))
			{
				char new_beated_pieces[10];
				for (int i = 0; i != beated_pieces_length; i++)
				{
					new_beated_pieces[i] = beated_pieces[i];
				}
				Board::GetAllKingBeatMovesCpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length, target_row, target_column, target_row + ind, target_column + ind, all_moves, all_moves_length);
			}
			else
				break;
		if (target_row - source_row > 0 && target_column - source_column < 0)
			if (Board::CanMoveToPosition(target_row + ind, target_column - ind, Board::ToPosition(piece_row, piece_column)))
			{
				char new_beated_pieces[10];
				for (int i = 0; i != beated_pieces_length; i++)
				{
					new_beated_pieces[i] = beated_pieces[i];
				}
				Board::GetAllKingBeatMovesCpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length, target_row, target_column, target_row + ind, target_column - ind, all_moves, all_moves_length);
			}
			else
				break;
		if (target_row - source_row < 0 && target_column - source_column > 0)
			if (Board::CanMoveToPosition(target_row - ind, target_column + ind, Board::ToPosition(piece_row, piece_column)))
			{
				char new_beated_pieces[10];
				for (int i = 0; i != beated_pieces_length; i++)
				{
					new_beated_pieces[i] = beated_pieces[i];
				}
				Board::GetAllKingBeatMovesCpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length, target_row, target_column, target_row - ind, target_column + ind, all_moves, all_moves_length);
			}
			else
				break;
		if (target_row - source_row < 0 && target_column - source_column < 0)
			if (Board::CanMoveToPosition(target_row - ind, target_column - ind, Board::ToPosition(piece_row, piece_column)))
			{
				char new_beated_pieces[10];
				for (int i = 0; i != beated_pieces_length; i++)
				{
					new_beated_pieces[i] = beated_pieces[i];
				}
				Board::GetAllKingBeatMovesCpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length, target_row, target_column, target_row - ind, target_column - ind, all_moves, all_moves_length);
			}
			else
				break;
	}
	if (!(target_row - source_row > 0 && target_column - source_column > 0))
		for (int ind = 1; ind < Board::size; ind++)
			if (Board::CanBeatPiece(target_row, target_column, target_row - ind, target_column - ind, Board::ToPosition(piece_row, piece_column)))
			{
				char beated_piece_position = Board::ToPosition(target_row - ind, target_column - ind);
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
					char new_beated_pieces[10];
					for (int i = 0; i != beated_pieces_length; i++)
					{
						new_beated_pieces[i] = beated_pieces[i];
					}
					new_beated_pieces[beated_pieces_length] = beated_piece_position;
					Board::GetAllKingBeatMovesCpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row - ind - 1, target_column - ind - 1, all_moves, all_moves_length);
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
				char beated_piece_position = Board::ToPosition(target_row + ind, target_column - ind);
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
					char new_beated_pieces[10];
					for (int i = 0; i != beated_pieces_length; i++)
					{
						new_beated_pieces[i] = beated_pieces[i];
					}
					new_beated_pieces[beated_pieces_length] = beated_piece_position;
					Board::GetAllKingBeatMovesCpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row + ind + 1, target_column - ind - 1, all_moves, all_moves_length);
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
				char beated_piece_position = Board::ToPosition(target_row - ind, target_column + ind);
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
					char new_beated_pieces[10];
					for (int i = 0; i != beated_pieces_length; i++)
					{
						new_beated_pieces[i] = beated_pieces[i];
					}
					new_beated_pieces[beated_pieces_length] = beated_piece_position;
					Board::GetAllKingBeatMovesCpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row - ind - 1, target_column + ind + 1, all_moves, all_moves_length);
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
				char beated_piece_position = Board::ToPosition(target_row + ind, target_column + ind);
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
					char new_beated_pieces[10];
					for (int i = 0; i != beated_pieces_length; i++)
					{
						new_beated_pieces[i] = beated_pieces[i];
					}
					new_beated_pieces[beated_pieces_length] = beated_piece_position;
					Board::GetAllKingBeatMovesCpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row + ind + 1, target_column + ind + 1, all_moves, all_moves_length);
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

__device__ void Board::GetPawnPossibleMovesGpu(char position, int &moves_count, Move *all_moves_device, int thread_id)
{
	char
		piece_row = Board::PositionToRow(position),
		piece_column = Board::PositionToColumn(position);
	switch (Board::player)
	{
	case Player::WHITE:
		if (Board::CanMoveToPosition(piece_row + 1, piece_column + 1, position))
			all_moves_device[1000 * (thread_id + 1) - 1 - moves_count++] = Move(position, Board::ToPosition(piece_row + 1, piece_column + 1), 0, 0);
		if (Board::CanMoveToPosition(piece_row + 1, piece_column - 1, position))
			all_moves_device[1000 * (thread_id + 1) - 1 - moves_count++] = Move(position, Board::ToPosition(piece_row + 1, piece_column - 1), 0, 0);
		break;
	case Player::BLACK:
		if (Board::CanMoveToPosition(piece_row - 1, piece_column + 1, position))
			all_moves_device[1000 * (thread_id + 1) - 1 - moves_count++] = Move(position, Board::ToPosition(piece_row - 1, piece_column + 1), 0, 0);
		if (Board::CanMoveToPosition(piece_row - 1, piece_column - 1, position))
			all_moves_device[1000 * (thread_id + 1) - 1 - moves_count++] = Move(position, Board::ToPosition(piece_row - 1, piece_column - 1), 0, 0);
		break;
	}

	//pr�ba bicia w czterech r�nych kierunkach
	if (Board::CanBeatPiece(piece_row, piece_column, piece_row - 1, piece_column - 1, position))
	{
		char beated_pieces[1];
		beated_pieces[0] = Board::ToPosition(piece_row - 1, piece_column - 1);
		Board::GetAllBeatMovesGpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row - 2, piece_column - 2, moves_count, all_moves_device, thread_id);
	}
	if (Board::CanBeatPiece(piece_row, piece_column, piece_row + 1, piece_column - 1, position))
	{
		char beated_pieces[1];
		beated_pieces[0] = Board::ToPosition(piece_row + 1, piece_column - 1);
		Board::GetAllBeatMovesGpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row + 2, piece_column - 2, moves_count, all_moves_device, thread_id);
	}
	if (Board::CanBeatPiece(piece_row, piece_column, piece_row - 1, piece_column + 1, position))
	{
		char beated_pieces[1];
		beated_pieces[0] = Board::ToPosition(piece_row - 1, piece_column + 1);
		Board::GetAllBeatMovesGpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row - 2, piece_column + 2, moves_count, all_moves_device, thread_id);
	}
	if (Board::CanBeatPiece(piece_row, piece_column, piece_row + 1, piece_column + 1, position))
	{
		char beated_pieces[1];
		beated_pieces[0] = Board::ToPosition(piece_row + 1, piece_column + 1);
		Board::GetAllBeatMovesGpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row + 2, piece_column + 2, moves_count, all_moves_device, thread_id);
	}
}

__device__ void Board::GetKingPossibleMovesGpu(char position, int &moves_count, Move *all_moves_device, int thread_id)
{
	char
		piece_row = Board::PositionToRow(position),
		piece_column = Board::PositionToColumn(position);

	//normalne ruchy w czterech kierunkach a� do napotkania pionka lub ko�ca planszy
	for (int i = 1; i < Board::size; i++)
	{
		if (Board::CanMoveToPosition(piece_row + i, piece_column + i, position))
			all_moves_device[1000 * (thread_id + 1) - 1 - moves_count++] = Move(position, Board::ToPosition(piece_row + i, piece_column + i), 0, 0);
		else
			break;
	}
	for (int i = 1; i < Board::size; i++)
	{
		if (Board::CanMoveToPosition(piece_row + i, piece_column - i, position))
			all_moves_device[1000 * (thread_id + 1) - 1 - moves_count++] = Move(position, Board::ToPosition(piece_row + i, piece_column - i), 0, 0);
		else
			break;
	}
	for (int i = 1; i < Board::size; i++)
	{
		if (Board::CanMoveToPosition(piece_row - i, piece_column + i, position))
			all_moves_device[1000 * (thread_id + 1) - 1 - moves_count++] = Move(position, Board::ToPosition(piece_row - i, piece_column + i), 0, 0);
		else
			break;
	}
	for (int i = 1; i < Board::size; i++)
	{
		if (Board::CanMoveToPosition(piece_row - i, piece_column - i, position))
			all_moves_device[1000 * (thread_id + 1) - 1 - moves_count++] = Move(position, Board::ToPosition(piece_row - i, piece_column - i), 0, 0);
		else
			break;
	}

	//pr�ba bicia w czterech r�nych kierunkach damk�
	for (int ind = 1; ind < Board::size; ind++)
		if (Board::CanBeatPiece(piece_row, piece_column, piece_row - ind, piece_column - ind, position))
		{
			char beated_pieces[1];
			beated_pieces[0] = Board::ToPosition(piece_row - ind, piece_column - ind);
			Board::GetAllKingBeatMovesGpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row - ind - 1, piece_column - ind - 1, moves_count, all_moves_device, thread_id);
		}
		else
		{
			if (!Board::CanMoveToPosition(piece_row - ind, piece_column - ind, position))
				break;
		}
	for (int ind = 1; ind < Board::size; ind++)
		if (Board::CanBeatPiece(piece_row, piece_column, piece_row - ind, piece_column + ind, position))
		{
			char beated_pieces[1];
			beated_pieces[0] = Board::ToPosition(piece_row - ind, piece_column + ind);
			Board::GetAllKingBeatMovesGpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row - ind - 1, piece_column + ind + 1, moves_count, all_moves_device, thread_id);
		}
		else
		{
			if (!Board::CanMoveToPosition(piece_row - ind, piece_column + ind, position))
				break;
		}
	for (int ind = 1; ind < Board::size; ind++)
		if (Board::CanBeatPiece(piece_row, piece_column, piece_row + ind, piece_column - ind, position))
		{
			char beated_pieces[1];
			beated_pieces[0] = Board::ToPosition(piece_row + ind, piece_column - ind);
			Board::GetAllKingBeatMovesGpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row + ind + 1, piece_column - ind - 1, moves_count, all_moves_device, thread_id);
		}
		else
		{
			if (!Board::CanMoveToPosition(piece_row + ind, piece_column - ind, position))
				break;
		}
	for (int ind = 1; ind < Board::size; ind++)
		if (Board::CanBeatPiece(piece_row, piece_column, piece_row + ind, piece_column + ind, position))
		{
			char beated_pieces[1];
			beated_pieces[0] = Board::ToPosition(piece_row + ind, piece_column + ind);
			Board::GetAllKingBeatMovesGpu(piece_row, piece_column, beated_pieces, 1, piece_row, piece_column, piece_row + ind + 1, piece_column + ind + 1, moves_count, all_moves_device, thread_id);
		}
		else
		{
			if (!Board::CanMoveToPosition(piece_row + ind, piece_column + ind, position))
				break;
		}
}

__device__ void Board::GetAllBeatMovesGpu(char piece_row, char piece_column, char *beated_pieces, char beated_pieces_length, char source_row, char source_column, char target_row, char target_column, int& moves_count, Move *all_moves_device, int thread_id)
{
	all_moves_device[1000 * (thread_id + 1) - 1 - moves_count++] = Move(Board::ToPosition(piece_row, piece_column), Board::ToPosition(target_row, target_column), beated_pieces_length, beated_pieces);
	if (Board::CanBeatPiece(target_row, target_column, target_row - 1, target_column - 1, Board::ToPosition(piece_row, piece_column)))
	{
		char beated_piece_position = Board::ToPosition(target_row - 1, target_column - 1);
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
			char new_beated_pieces[10];
			for (int i = 0; i != beated_pieces_length; i++)
			{
				new_beated_pieces[i] = beated_pieces[i];
			}
			new_beated_pieces[beated_pieces_length] = beated_piece_position;
			Board::GetAllBeatMovesGpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row - 2, target_column - 2, moves_count, all_moves_device, thread_id);
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
			char new_beated_pieces[10];
			for (int i = 0; i != beated_pieces_length; i++)
			{
				new_beated_pieces[i] = beated_pieces[i];
			}
			new_beated_pieces[beated_pieces_length] = beated_piece_position;
			Board::GetAllBeatMovesGpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row + 2, target_column - 2, moves_count, all_moves_device, thread_id);
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
			char new_beated_pieces[10];
			for (int i = 0; i != beated_pieces_length; i++)
			{
				new_beated_pieces[i] = beated_pieces[i];
			}
			new_beated_pieces[beated_pieces_length] = beated_piece_position;
			Board::GetAllBeatMovesGpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row - 2, target_column + 2, moves_count, all_moves_device, thread_id);
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
			char new_beated_pieces[10];
			for (int i = 0; i != beated_pieces_length; i++)
			{
				new_beated_pieces[i] = beated_pieces[i];
			}
			new_beated_pieces[beated_pieces_length] = beated_piece_position;
			Board::GetAllBeatMovesGpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row + 2, target_column + 2, moves_count, all_moves_device, thread_id);
		}
	}
}

__device__ void Board::GetAllKingBeatMovesGpu(char piece_row, char piece_column, char *beated_pieces, char beated_pieces_length, char source_row, char source_column, char target_row, char target_column, int& moves_count, Move *all_moves_device, int thread_id)
{
	//sprawdzanie czy ruch zosta� ju� dodany
	bool flag = true;
	Move new_move = Move(Board::ToPosition(piece_row, piece_column), Board::ToPosition(target_row, target_column), beated_pieces_length, beated_pieces);
	for (int i = 0; i != moves_count; i++)
	{
		if (
			all_moves_device[1000 * (thread_id + 1) - 1 - i].new_position == new_move.new_position &&
			all_moves_device[1000 * (thread_id + 1) - 1 - i].old_position == new_move.old_position &&
			all_moves_device[1000 * (thread_id + 1) - 1 - i].beated_pieces_count == new_move.beated_pieces_count
			)
		{
			if (new_move.beated_pieces_count == 0)
			{
				flag = false;
				break;
			}
			for (int j = 0; j != beated_pieces_length; j++)
			{
				if (all_moves_device[1000 * (thread_id + 1) - 1 - i].beated_pieces[j] == new_move.beated_pieces[j])
				{
					if (j + 1 == beated_pieces_length)
					{
						flag = false;
					}
				}
				if (!flag)
					break;
			}
		}
	}
	if (flag)
		all_moves_device[1000 * (thread_id + 1) - 1 - moves_count++] = new_move;
	for (int ind = 1; ind < Board::size; ind++)
	{
		if (target_row - source_row > 0 && target_column - source_column > 0)
			if (Board::CanMoveToPosition(target_row + ind, target_column + ind, Board::ToPosition(piece_row, piece_column)))
			{
				char new_beated_pieces[10];
				for (int i = 0; i != beated_pieces_length; i++)
				{
					new_beated_pieces[i] = beated_pieces[i];
				}
				Board::GetAllKingBeatMovesGpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length, target_row, target_column, target_row + ind, target_column + ind, moves_count, all_moves_device, thread_id);
			}
			else
				break;
		if (target_row - source_row > 0 && target_column - source_column < 0)
			if (Board::CanMoveToPosition(target_row + ind, target_column - ind, Board::ToPosition(piece_row, piece_column)))
			{
				char new_beated_pieces[10];
				for (int i = 0; i != beated_pieces_length; i++)
				{
					new_beated_pieces[i] = beated_pieces[i];
				}
				Board::GetAllKingBeatMovesGpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length, target_row, target_column, target_row + ind, target_column - ind, moves_count, all_moves_device, thread_id);
			}
			else
				break;
		if (target_row - source_row < 0 && target_column - source_column > 0)
			if (Board::CanMoveToPosition(target_row - ind, target_column + ind, Board::ToPosition(piece_row, piece_column)))
			{
				char new_beated_pieces[10];
				for (int i = 0; i != beated_pieces_length; i++)
				{
					new_beated_pieces[i] = beated_pieces[i];
				}
				Board::GetAllKingBeatMovesGpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length, target_row, target_column, target_row - ind, target_column + ind, moves_count, all_moves_device, thread_id);
			}
			else
				break;
		if (target_row - source_row < 0 && target_column - source_column < 0)
			if (Board::CanMoveToPosition(target_row - ind, target_column - ind, Board::ToPosition(piece_row, piece_column)))
			{
				char new_beated_pieces[10];
				for (int i = 0; i != beated_pieces_length; i++)
				{
					new_beated_pieces[i] = beated_pieces[i];
				}
				Board::GetAllKingBeatMovesGpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length, target_row, target_column, target_row - ind, target_column - ind, moves_count, all_moves_device, thread_id);
			}
			else
				break;
	}
	if (!(target_row - source_row > 0 && target_column - source_column > 0))
		for (int ind = 1; ind < Board::size; ind++)
			if (Board::CanBeatPiece(target_row, target_column, target_row - ind, target_column - ind, Board::ToPosition(piece_row, piece_column)))
			{
				char beated_piece_position = Board::ToPosition(target_row - ind, target_column - ind);
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
					char new_beated_pieces[10];
					for (int i = 0; i != beated_pieces_length; i++)
					{
						new_beated_pieces[i] = beated_pieces[i];
					}
					new_beated_pieces[beated_pieces_length] = beated_piece_position;
					Board::GetAllKingBeatMovesGpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row - ind - 1, target_column - ind - 1, moves_count, all_moves_device, thread_id);
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
				char beated_piece_position = Board::ToPosition(target_row + ind, target_column - ind);
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
					char new_beated_pieces[10];
					for (int i = 0; i != beated_pieces_length; i++)
					{
						new_beated_pieces[i] = beated_pieces[i];
					}
					new_beated_pieces[beated_pieces_length] = beated_piece_position;
					Board::GetAllKingBeatMovesGpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row + ind + 1, target_column - ind - 1, moves_count, all_moves_device, thread_id);
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
				char beated_piece_position = Board::ToPosition(target_row - ind, target_column + ind);
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
					char new_beated_pieces[10];
					for (int i = 0; i != beated_pieces_length; i++)
					{
						new_beated_pieces[i] = beated_pieces[i];
					}
					new_beated_pieces[beated_pieces_length] = beated_piece_position;
					Board::GetAllKingBeatMovesGpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row - ind - 1, target_column + ind + 1, moves_count, all_moves_device, thread_id);
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
				char beated_piece_position = Board::ToPosition(target_row + ind, target_column + ind);
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
					char new_beated_pieces[10];
					for (int i = 0; i != beated_pieces_length; i++)
					{
						new_beated_pieces[i] = beated_pieces[i];
					}
					new_beated_pieces[beated_pieces_length] = beated_piece_position;
					Board::GetAllKingBeatMovesGpu(piece_row, piece_column, new_beated_pieces, beated_pieces_length + 1, target_row, target_column, target_row + ind + 1, target_column + ind + 1, moves_count, all_moves_device, thread_id);
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

__device__ __host__ bool Board::CanMoveToPosition(char position_row, char position_column, char source_move_position)
{
	int position = Board::ToPosition(position_row, position_column);
	return
		position_row >= 0 &&
		position_column >= 0 &&
		position_row <= Board::size - 1 &&
		position_column <= Board::size - 1 &&
		(
			Board::pieces[position] == 0 ||
			position == source_move_position //bij�cy pionek nie powinien by� brany pod uwag�
			);
}

__device__ __host__ bool Board::CanBeatPiece(char position_row, char position_column, char target_piece_position_row, char target_piece_position_column, char source_move_position)
{
	char
		target_piece_position = ToPosition(target_piece_position_row, target_piece_position_column),
		row_after_beat = target_piece_position_row + (target_piece_position_row - position_row > 0 ? 1 : -1),
		column_after_beat = target_piece_position_column + (target_piece_position_column - position_column > 0 ? 1 : -1);

	//sprawdzenie czy bite pole i pole po biciu mieszcz� si� w planszy
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

	//sprawdzenie czy jest przeciwny pionek na pozycji i czy po biciu mo�na postawi� pionka na nast�pnym polu
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

__device__ __host__ char Board::PositionToColumn(char position)
{
	int div = size / 2;
	return 2 * ((position - 1) % div) + (Board::PositionToRow(position) % 2 == 1 ? 1 : 0);
}

__device__ __host__ char Board::PositionToRow(char position)
{
	int div = Board::size / 2;
	return Board::size - 1 - (position - 1) / div;
}

__device__ __host__ char Board::ToPosition(char row, char column)
{
	if ((column % 2 == 0 && row % 2 == 1) || (row % 2 == 0 && column % 2 == 1))
		return -1;
	else
		return size / 2 * (size - row - 1) + ((row % 2 == 0) ? 1 : 0) + (column + 1) / 2;
}

__host__ Player Board::RolloutCpu()
{
	int
		moves_count = 0,
		move_ind = 0;
	Board current_board = *this;
	while (1)
	{
		if (current_board.IsGameFinished())
			break;
		Move *possible_moves = current_board.GetPossibleMovesCpu(moves_count);
		if (moves_count == 0)
			break;

		move_ind = rand() % moves_count;
		Board new_board = current_board.GetBoardAfterMove(possible_moves[move_ind]);
		delete[] possible_moves;
		current_board = new_board;

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

__device__ Player Board::RolloutGpu(curandState *state, Move *all_moves_device, int thread_id)
{
	int
		moves_count = 0,
		move_ind = 0;
	Board current_board = *this;
	while (1)
	{
		if (current_board.IsGameFinished())
			break;
		current_board.GetPossibleMovesGpu(moves_count, all_moves_device, thread_id);
		if (moves_count == 0)
			break;
		move_ind = GenerateRandomInt(state, 0, moves_count - 1);
		//move_ind = rand() % moves_count;
		move_ind += 1000 * thread_id;
		Board new_board = current_board.GetBoardAfterMove(all_moves_device[move_ind]);
		current_board = new_board;

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

__device__ int Board::GenerateRandomInt(curandState *state, int min, int max)
{
	float rand = curand_uniform(state);
	rand *= (max - min + 0.999999);
	rand += min;
	return (int)truncf(rand);
}

__device__ __host__ bool Board::IsGameFinished()
{
	char player_pieces = 0;
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