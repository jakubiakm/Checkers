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
        public EngineKind Kind
        {
            get
            {
                return EngineKind.Human;
            }
        }

        public PieceColor Color { get; set; }

        public List<Piece> HumanMove { get; set; }

        public HumanEngine(PieceColor color)
        {
            Color = color;
        }

        public void Reset()
        {

        }

        public Move MakeMove(CheckersBoard currentBoard, GameVariant variant, List<Move> gameMoves)
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
                var possibleMoves = allPossibleMoves.Where(m => m.OldPiece.Position == HumanMove[0].Position && m.NewPiece.Position == HumanMove.Last().Position).ToList();
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
            int fromNumber = move.OldPiece.Position;
            int toNumber = move.NewPiece.Position;

            if (move.BeatedPieces == null)
                return $"({fromNumber}-{toNumber})";
            else
            {
                string numberString = "(";
                foreach (var piece in move.BeatedPieces)
                {
                    numberString += $"{Piece.ToPosition(piece.BeatPieceRow, piece.BeatPieceColumn, 10)}x";
                }
                numberString += move.NewPiece.Position;
                numberString += ")";
                return numberString;
            }
        }
    }
}
