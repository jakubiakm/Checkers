#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "move.cuh"

enum Player
{
	WHITE = 0, 
	BLACK = 1
};

class Board
{
public:
	char size;
	char *pieces;
	Player player;

	__device__ __host__ Board(char size, char* _pieces, Player player);
	__device__ __host__ Board();
	__device__ __host__ ~Board();
	__host__ Move* GetPossibleMovesCpu(int &moves_count);
	__device__ Move* GetPossibleMovesGpu(int &moves_count);
	__device__ __host__ Board GetBoardAfterMove(Move move);
	__host__ Player RolloutCpu();
	__device__ Player RolloutGpu();
private:
	__host__ Move* GetPawnPossibleMovesCpu(char position, int &moves_count);
	__host__ Move* GetKingPossibleMovesCpu(char position, int &moves_count);
	__host__ void GetAllBeatMovesCpu(char piece_row, char piece_column, char *beated_pieces, char beated_pieces_length, char source_row, char source_column, char target_row, char target_column, Move* all_moves, int& all_moves_length);
	__host__ void GetAllKingBeatMovesCpu(char piece_row, char piece_column, char *beated_pieces, char beated_pieces_length, char source_row, char source_column, char target_row, char target_column, Move* all_moves, int& all_moves_length);
	__device__ Move* GetPawnPossibleMovesGpu(char position, int &moves_count);
	__device__ Move* GetKingPossibleMovesGpu(char position, int &moves_count);
	__device__ void GetAllBeatMovesGpu(char piece_row, char piece_column, char *beated_pieces, char beated_pieces_length, char source_row, char source_column, char target_row, char target_column, Move* all_moves, int& all_moves_length);
	__device__ void GetAllKingBeatMovesGpu(char piece_row, char piece_column, char *beated_pieces, char beated_pieces_length, char source_row, char source_column, char target_row, char target_column, Move* all_moves, int& all_moves_length);
	__device__ __host__ bool CanMoveToPosition(char position_row, char position_column, char source_move_position);
	__device__ __host__ bool CanBeatPiece(char position_row, char position_column, char target_piece_position_row, char target_piece_position_column, char source_move_position);
	__device__ __host__ char PositionToRow(char position);
	__device__ __host__ char PositionToColumn(char position);
	__device__ __host__ char ToPosition(char row, char column);
	__device__ __host__ bool IsGameFinished();
};