#pragma once
class Move
{
public:
	int beatedPiecesCount;
	int *beatedPieces;
	int oldPosition;
	int newPosition;
	Move();
	Move(int oldPosition, int newPosition, int beatedPiecesCount, int *beatedPieces);
	~Move();
};

