#include "Move.h"

Move::Move(int oldPosition, int newPosition, int beatedPiecesCount, int *beatedPieces) : oldPosition(oldPosition), newPosition(newPosition), beatedPiecesCount(beatedPiecesCount), beatedPieces(beatedPieces)
{
}

Move::Move()
{
}

Move::~Move()
{
}
