#include "Move.h"



Move::Move(int oldPositiion, int newPosition, int beatedPiecesCount, int *beatedPieces)
{
	OldPosition = oldPositiion;
	NewPosition = newPosition;
	BeatedPiecesCount = beatedPiecesCount;
	BeatedPieces = beatedPieces;
}

Move::Move()
{

}


Move::~Move()
{
}
