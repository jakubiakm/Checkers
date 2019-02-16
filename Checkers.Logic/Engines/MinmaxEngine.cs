using Checkers.Logic.AlgorithmObjects;
using Checkers.Logic.Enums;
using Checkers.Logic.Exceptions;
using Checkers.Logic.GameObjects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.Engines
{
    public class MinMaxEngine : IEngine
    {
        public int MinmaxTreeDepth { get; set; }

        public EngineKind Kind
        {
            get
            {
                return EngineKind.MinMax;
            }
        }

        public PieceColor Color { get; set; }

        private Random randomGenerator;

        public MinMaxEngine(PieceColor color, int treeDepth)
        {
            MinmaxTreeDepth = treeDepth;
            randomGenerator = new Random();
            Color = color;
        }

        public void Reset()
        {
            randomGenerator = new Random();
        }

        public Move MakeMove(CheckersBoard currentBoard, GameVariant variant)
        {
            List<Move> allPossibleMoves = currentBoard.GetAllPossibleMoves(Color);
            int count = allPossibleMoves.Count;
            if (count == 0)
                throw new NotAvailableMoveException(Color);
            if (count == 1)
            {
                return allPossibleMoves.First();
            }
            else
            {
                MinMaxTree tree = new MinMaxTree(MinmaxTreeDepth);
                tree.BuildTree(currentBoard, Color);
                int elemIndex = tree.ChooseBestMove();
                return allPossibleMoves[elemIndex];
            }
        }
    }
}
