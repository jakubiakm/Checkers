using Checkers.Logic.Engines;
using Checkers.Logic.Enums;
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

        public Game(IEngine whiteEngine, IEngine blackEngine, int boardSize = 10, int numberOfWhitePieces = 20, int numberOfBlackPieces = 20)
        {
            WhitePlayerEngine = whiteEngine;
            BlackPlayerEngine = blackEngine;
            Board = new CheckersBoard(boardSize, numberOfWhitePieces, numberOfBlackPieces);
        }

        public void MakeMove(PieceColor color)
        {
            History.Add(Board.DeepClone());
            switch(color)
            {
                case PieceColor.White:
                    Board.MakeMove(WhitePlayerEngine.MakeMove(Board));
                    break;
                case PieceColor.Black:
                    Board.MakeMove(BlackPlayerEngine.MakeMove(Board));
                    break;
            }
        }
    }
}
