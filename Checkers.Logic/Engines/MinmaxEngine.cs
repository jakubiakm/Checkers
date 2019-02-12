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
    public class MinmaxEngine : IEngine
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

        public MinmaxEngine(PieceColor color, int treeDepth)
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
            int elemIndex = randomGenerator.Next(count);
            return allPossibleMoves[elemIndex];
        }
    }
}
