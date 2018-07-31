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

        public Move LastMove { get; private set; }

        public Game(IEngine whiteEngine, IEngine blackEngine, int boardSize = 10, int numberOfWhitePieces = 20, int numberOfBlackPieces = 20)
        {
            WhitePlayerEngine = whiteEngine;
            BlackPlayerEngine = blackEngine;
            Board = new CheckersBoard(boardSize, numberOfWhitePieces, numberOfBlackPieces);
            History = new List<CheckersBoard>();
        }

        public Game(IEngine whiteEngine, IEngine blackEngine, int boardSize, List<Piece> pieces)
        {
            WhitePlayerEngine = whiteEngine;
            BlackPlayerEngine = blackEngine;
            Board = new CheckersBoard(boardSize, pieces);
            History = new List<CheckersBoard>();
        }

        public Move MakeMove(PieceColor color)
        {
            History.Add(Board.DeepClone());
            switch(color)
            {
                case PieceColor.White:
                    LastMove = Board.MakeMove(WhitePlayerEngine.MakeMove(Board));
                    break;
                case PieceColor.Black:
                    LastMove = Board.MakeMove(BlackPlayerEngine.MakeMove(Board));
                    break;
            }
            if (Board.PiecesOnBoard.Where(p => p.Color == PieceColor.Black).Count() == 0)
                throw new NoAvailablePiecesException(PieceColor.Black, LastMove);
            if (Board.PiecesOnBoard.Where(p => p.Color == PieceColor.White).Count() == 0)
                throw new NoAvailablePiecesException(PieceColor.White, LastMove);
            return LastMove;
        }
    }
}
