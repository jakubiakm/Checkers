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
	int size;
	int* pieces;
	Player player;

	__device__ __host__ Board(int size, int* _pieces, Player player);
	__device__ __host__ Board();
	__device__ __host__ ~Board();
	__device__ __host__ Move* GetPossibleMoves(int &moves_count);
	__device__ __host__ Board GetBoardAfterMove(Move move);
	
private:
	__device__ __host__ Move* GetPawnPossibleMoves(int position, int &moves_count);
	__device__ __host__ Move* GetKingPossibleMoves(int position, int &moves_count);
	__device__ __host__ void GetAllBeatMoves(int piece_row, int piece_column, int *beated_pieces, int beated_pieces_length, int source_row, int source_column, int target_row, int target_column, Move* all_moves, int& all_moves_length);
	__device__ __host__ void GetAllKingBeatMoves(int piece_row, int piece_column, int *beated_pieces, int beated_pieces_length, int source_row, int source_column, int target_row, int target_column, Move* all_moves, int& all_moves_length);
	__device__ __host__ bool CanMoveToPosition(int position_row, int position_column, int source_move_position);
	__device__ __host__ bool CanBeatPiece(int position_row, int position_column, int target_piece_position_row, int target_piece_position_column, int source_move_position);
	__device__ __host__ int PositionToRow(int position);
	__device__ __host__ int PositionToColumn(int position);
	__device__ __host__ int ToPosition(int row, int column);
};