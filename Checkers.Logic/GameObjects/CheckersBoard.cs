using Checkers.Logic.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.GameObjects
{
    [Serializable]
    public class CheckersBoard
    {
        public int Size { get; private set; } = 10;

        public List<Piece> PiecesOnBoard { get; private set; }

        public CheckersBoard(int size, int numberOfWhitePieces, int numberOfBlackPieces)
        {
            PiecesOnBoard = new List<Piece>();
            int row = 0;
            int column = 0;
            if (size * size / 2 < numberOfWhitePieces + numberOfBlackPieces)
                throw new ArgumentException("Pionki nie mieszczą się na planszy");
            if (size % 2 == 1)
                throw new ArgumentException("Rozmiar planszy musi być liczbą parzystą");
            while (numberOfWhitePieces-- > 0)
            {
                if (row >= size)
                {
                    column++;
                    row = column % 2 == 0 ? 0 : 1;
                }
                PiecesOnBoard.Add(new Piece(row, column, PieceColor.White));
                row += 2;
            }
            row = size - 1;
            column = size - 1;
            while (numberOfBlackPieces-- > 0)
            {
                if(row < 0)
                {
                    column--;
                    row = column % 2 == 0 ? size - 2 : size - 1;
                }
                PiecesOnBoard.Add(new Piece(row, column, PieceColor.Black));
                row -= 2;
            }
        }

        public List<Move> GetPossibleMoves(PieceColor color)
        {
            throw new NotImplementedException();
        }

        public void MakeMove(Move move)
        {
            PiecesOnBoard[PiecesOnBoard.FindIndex(p => p.Row == move.OldPiece.Row && p.Column == move.OldPiece.Column)] = move.NewPiece;
            foreach (var piece in move.BeatedPieces ?? new List<Piece>())
                PiecesOnBoard.Remove(piece);
        }
    }
}
