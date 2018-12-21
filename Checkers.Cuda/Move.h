#pragma once
class Move
{
public:
	int BeatedPiecesCount;
	int *BeatedPieces;
	int OldPosition;
	int NewPosition;
	Move();
	Move(int oldPositiion, int newPosition, int beatedPiecesCount, int *beatedPieces);
	~Move();
};

