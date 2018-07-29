using Checkers.Logic.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
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
            Size = size;
            int row = 0;
            int column = 0;
            if (size * size / 2 < numberOfWhitePieces + numberOfBlackPieces)
                throw new ArgumentException("Pionki nie mieszczą się na planszy");
            if (size % 2 == 1)
                throw new ArgumentException("Rozmiar planszy musi być liczbą parzystą");
            while (numberOfWhitePieces-- > 0)
            {
                if (column >= size)
                {
                    row++;
                    column = row % 2 == 0 ? 0 : 1;
                }
                PiecesOnBoard.Add(new Piece(row, column, PieceColor.White, false));
                column += 2;
            }
            row = size - 1;
            column = size - 1;
            while (numberOfBlackPieces-- > 0)
            {
                if (column < 0)
                {
                    row--;
                    column = row % 2 == 0 ? size - 2 : size - 1;
                }
                PiecesOnBoard.Add(new Piece(row, column, PieceColor.Black, false));
                column -= 2;
            }
        }

        public List<Move> GetAllPossibleMoves(PieceColor color)
        {
            List<Move> possibleMoves = new List<Move>();
            PiecesOnBoard.Where(p => p.Color == color).ToList().ForEach(p => possibleMoves.AddRange(GetPiecePossibleMoves(p)));
            var maximumBeatedPieces = possibleMoves.Count == 0 ? 0 : possibleMoves.Max(m => m.BeatedPieces?.Count ?? 0);
            return possibleMoves.Where(m => (m.BeatedPieces?.Count ?? 0) == maximumBeatedPieces)?.ToList() ?? new List<Move>();
        }

        public void MakeMove(Move move)
        {
            PiecesOnBoard[PiecesOnBoard.FindIndex(p => p.Row == move.OldPiece.Row && p.Column == move.OldPiece.Column)] = move.NewPiece;
            foreach (var piece in move.BeatedPieces ?? new List<Piece>())
                PiecesOnBoard.Remove(piece);
        }

        private List<Move> GetPiecePossibleMoves(Piece piece)
        {
            List<Move> possibleMoves = new List<Move>();
            //normalne ruchy do przodu
            switch (piece.Color)
            {
                case PieceColor.White:
                    if (CanMoveToPosition(piece.Row + 1, piece.Column + 1))
                        possibleMoves.Add(new Move(piece, new Piece(piece.Row + 1, piece.Column + 1, PieceColor.White, piece.Row + 1 == Size - 1), null));
                    if (CanMoveToPosition(piece.Row + 1, piece.Column - 1))
                        possibleMoves.Add(new Move(piece, new Piece(piece.Row + 1, piece.Column - 1, PieceColor.White, piece.Row + 1 == Size - 1), null));
                    break;
                case PieceColor.Black:
                    if (CanMoveToPosition(piece.Row - 1, piece.Column + 1))
                        possibleMoves.Add(new Move(piece, new Piece(piece.Row - 1, piece.Column + 1, PieceColor.Black, piece.Row - 1 == 0), null));
                    if (CanMoveToPosition(piece.Row - 1, piece.Column - 1))
                        possibleMoves.Add(new Move(piece, new Piece(piece.Row - 1, piece.Column - 1, PieceColor.Black, piece.Row - 1 == 0), null));
                    break;
            }
            //próba bicia w czterech różnych kierunkach
            if (CanBeatPiece(piece, piece.Row - 1, piece.Column - 1))
            {
                Piece beatedPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == piece.Row - 1 && p.Column == piece.Column - 1);
                GetAllBeatMoves(piece, new List<Piece>() { beatedPiece }, piece.Row, piece.Column, piece.Row - 2, piece.Column - 2, ref possibleMoves);
            }
            if (CanBeatPiece(piece, piece.Row + 1, piece.Column - 1))
            {
                Piece beatedPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == piece.Row + 1 && p.Column == piece.Column - 1);
                GetAllBeatMoves(piece, new List<Piece>() { beatedPiece }, piece.Row, piece.Column, piece.Row + 2, piece.Column - 2, ref possibleMoves);
            }
            if (CanBeatPiece(piece, piece.Row - 1, piece.Column + 1))
            {
                Piece beatedPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == piece.Row - 1 && p.Column == piece.Column + 1);
                GetAllBeatMoves(piece, new List<Piece>() { beatedPiece }, piece.Row, piece.Column, piece.Row - 2, piece.Column + 2, ref possibleMoves);
            }
            if (CanBeatPiece(piece, piece.Row + 1, piece.Column + 1))
            {
                Piece beatedPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == piece.Row + 1 && p.Column == piece.Column + 1);
                GetAllBeatMoves(piece, new List<Piece>() { beatedPiece }, piece.Row, piece.Column, piece.Row + 2, piece.Column + 2, ref possibleMoves);
            }
            return possibleMoves;
        }
        private bool CanMoveToPosition(int row, int column)
        {
            return
                row >= 0 && row < Size && column >= 0 && column < Size &&
                PiecesOnBoard.Where(p => p.Row == row && p.Column == column).Count() == 0;
        }

        private bool CanBeatPiece(Piece piece, int row, int column)
        {
            int rowAfterBeat = row + (row - piece.Row);
            int columnAfterBeat = column + (column - piece.Column);
            //sprawdzenie czy bite pole i pole po biciu mieszczą się w planszy
            if (!(row >= 0 && row < Size && column >= 0 && column < Size &&
                rowAfterBeat >= 0 && rowAfterBeat < Size && columnAfterBeat >= 0 && columnAfterBeat < Size))
                return false;
            //sprawdzenie czy jest przeciwny pionek na pozycji i czy po biciu można postawić pionka na następnym polu
            if (PiecesOnBoard.Where(p => p.Row == row && p.Column == column && p.Color != piece.Color).Count() > 0 &&
                PiecesOnBoard.Where(p => p.Row == rowAfterBeat && p.Column == columnAfterBeat).Count() == 0)
                return true;
            return false;
        }

        private void GetAllBeatMoves(Piece piece, List<Piece> beatedPieces, int sourceRow, int sourceColumn, int targetRow, int targetColumn, ref List<Move> allMoves)
        {
            allMoves.Add(new Move(piece, new Piece(targetRow, targetColumn, piece.Color, piece.Color == PieceColor.White ? targetRow == Size - 1 : targetRow == 0), beatedPieces));
            if (CanBeatPiece(piece, targetRow - 1, targetColumn - 1))
            {
                Piece beatedPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == targetRow - 1 && p.Column == targetColumn - 1);
                if (beatedPieces.Where(p => p.Row == beatedPiece.Row && p.Column == beatedPiece.Column && p.Color == beatedPiece.Color).Count() == 0)
                {
                    beatedPieces.Add(beatedPiece);
                    GetAllBeatMoves(piece, beatedPieces, targetRow, targetColumn, targetRow - 2, targetColumn - 2, ref allMoves);
                }
            }
            if (CanBeatPiece(piece, targetRow + 1, targetColumn - 1))
            {
                Piece beatedPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == targetRow + 1 && p.Column == targetColumn - 1);
                if (beatedPieces.Where(p => p.Row == beatedPiece.Row && p.Column == beatedPiece.Column && p.Color == beatedPiece.Color).Count() == 0)
                {
                    beatedPieces.Add(beatedPiece);
                    GetAllBeatMoves(piece, beatedPieces, targetRow, targetColumn, targetRow + 2, targetColumn - 2, ref allMoves);
                }
            }
            if (CanBeatPiece(piece, targetRow - 1, targetColumn + 1))
            {
                Piece beatedPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == targetRow - 1 && p.Column == targetColumn + 1);
                if (beatedPieces.Where(p => p.Row == beatedPiece.Row && p.Column == beatedPiece.Column && p.Color == beatedPiece.Color).Count() == 0)
                {
                    beatedPieces.Add(beatedPiece);
                    GetAllBeatMoves(piece, beatedPieces, targetRow, targetColumn, targetRow - 2, targetColumn + 2, ref allMoves);
                }
            }
            if (CanBeatPiece(piece, targetRow + 1, targetColumn + 1))
            {
                Piece beatedPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == targetRow + 1 && p.Column == targetColumn + 1);
                if (beatedPieces.Where(p => p.Row == beatedPiece.Row && p.Column == beatedPiece.Column && p.Color == beatedPiece.Color).Count() == 0)
                {
                    beatedPieces.Add(beatedPiece);
                    GetAllBeatMoves(piece, beatedPieces, targetRow, targetColumn, targetRow + 2, targetColumn + 2, ref allMoves);
                }
            }
        }
    }
}
