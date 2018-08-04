using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Checkers.Logic.Enums;
using Checkers.Logic.Exceptions;
using Checkers.Logic.GameObjects;

namespace Checkers.Logic.Engines
{
    public class HumanEngine : IEngine
    {
        public PieceColor Color { get; set; }

        public List<Piece> HumanMove { get; set; }

        public HumanEngine(PieceColor color)
        {
            Color = color;
        }
        public Move MakeMove(CheckersBoard currentBoard)
        {
            List<Move> allPossibleMoves = currentBoard.GetAllPossibleMoves(Color);
            int count = allPossibleMoves.Count;
            if (count == 0)
                throw new NotAvailableMoveException(Color);
            StringBuilder stringBuilder = new StringBuilder();
            foreach(var move in allPossibleMoves)
            {
                stringBuilder.Append($"{GetPossibility(move)}, ");
            }
            stringBuilder.Remove(stringBuilder.Length - 2, 2);
            int beatedPawns = allPossibleMoves[0].BeatedPieces?.Count ?? 1;
            if(HumanMove.Count == 1 + beatedPawns)
            {
                var possibleMoves = allPossibleMoves.Where(m => m.OldPiece.Row == HumanMove[0].Row && m.OldPiece.Column == HumanMove[0].Column && m.NewPiece.Row == HumanMove.Last().Row && m.NewPiece.Column == HumanMove.Last().Column).ToList();
                HumanMove.Remove(HumanMove.First());
                HumanMove.Remove(HumanMove.Last());
                for (int i = 0; i < HumanMove.Count; i++)
                {
                    possibleMoves = possibleMoves.Where(p => p.BeatedPieces[i + 1].BeatPieceColumn == HumanMove[i].Column && p.BeatedPieces[i + 1].BeatPieceRow == HumanMove[i].Row).ToList();
                }
                var humanMove = possibleMoves.SingleOrDefault();
                if (humanMove == null)
                    throw new WrongMoveException(allPossibleMoves[0].BeatedPieces == null ? 0 : beatedPawns, stringBuilder.ToString());
                return humanMove;
            }
            else
            {
                throw new WrongMoveException(allPossibleMoves[0].BeatedPieces == null ? 0 : beatedPawns, stringBuilder.ToString());
            }
        }

        private string GetPossibility(Move move)
        {
            int fromNumber = GetCheckersPositionNumber(10, move.OldPiece.Row, move.OldPiece.Column);
            int toNumber = GetCheckersPositionNumber(10, move.NewPiece.Row, move.NewPiece.Column);

            if (move.BeatedPieces == null)
                return $"({fromNumber}-{toNumber})";
            else
            {
                string numberString = "(";
                foreach (var piece in move.BeatedPieces)
                {
                    numberString += $"{GetCheckersPositionNumber(10, piece.BeatPieceRow, piece.BeatPieceColumn)}x";
                }
                numberString += GetCheckersPositionNumber(10, move.NewPiece.Row, move.NewPiece.Column);
                numberString += ")";
                return numberString;
            }
        }

        private int GetCheckersPositionNumber(int size, int row, int column)
        {
            return size / 2 * (size - row - 1) + ((row % 2 == 0) ? 1 : 0) + (column + 1) / 2;
        }
    }
}
