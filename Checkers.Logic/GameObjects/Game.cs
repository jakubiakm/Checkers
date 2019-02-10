using Checkers.Logic.Engines;
using Checkers.Logic.Enums;
using Checkers.Logic.Exceptions;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.GameObjects
{
    public class Game
    {
        public List<CheckersBoard> History { get; private set; }

        public CheckersBoard Board { get; private set; }

        public IEngine WhitePlayerEngine { get; private set; }

        public IEngine BlackPlayerEngine { get; private set; }

        public GameVariant Variant { get; set; }

        public Game(IEngine whiteEngine, IEngine blackEngine, int boardSize, int numberOfWhitePieces, int numberOfBlackPieces, GameVariant variant)
        {
            WhitePlayerEngine = whiteEngine;
            BlackPlayerEngine = blackEngine;
            Board = new CheckersBoard(boardSize, numberOfWhitePieces, numberOfBlackPieces);
            History = new List<CheckersBoard>();
            Variant = variant;
        }

        public Game(IEngine whiteEngine, IEngine blackEngine, int boardSize, List<Piece> pieces, GameVariant variant)
        {
            WhitePlayerEngine = whiteEngine;
            BlackPlayerEngine = blackEngine;
            Board = new CheckersBoard(boardSize, pieces);
            History = new List<CheckersBoard>();
            Variant = variant;
        }

        public Move MakeMove(PieceColor color)
        {
            var x = Board.DeepClone();
            switch(color)
            {
                case PieceColor.White:
                    Board.LastMove = Board.MakeMove(WhitePlayerEngine.MakeMove(Board, Variant));
                    break;
                case PieceColor.Black:
                    Board.LastMove = Board.MakeMove(BlackPlayerEngine.MakeMove(Board, Variant));
                    break;
            }
            if (Board.PiecesOnBoard.Where(p => p.Color == PieceColor.Black).Count() == 0)
                throw new NoAvailablePiecesException(PieceColor.Black, Board.LastMove);
            if (Board.PiecesOnBoard.Where(p => p.Color == PieceColor.White).Count() == 0)
                throw new NoAvailablePiecesException(PieceColor.White, Board.LastMove);
            History.Add(x);
            return Board.LastMove;
        }
    }
}
