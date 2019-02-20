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
    public class AlphaBetaEngine : IEngine
    {
        public int AlphaBetaTreeDepth { get; set; }

        public EngineKind Kind
        {
            get
            {
                return EngineKind.AlphaBeta;
            }
        }

        public PieceColor Color { get; set; }

        private Random randomGenerator;

        public AlphaBetaEngine(PieceColor color, int treeDepth)
        {
            AlphaBetaTreeDepth = treeDepth;
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
                AlphaBetaTree tree = new AlphaBetaTree(AlphaBetaTreeDepth, Color, currentBoard);
                int elemIndex = tree.ChooseBestMove(variant);
                return allPossibleMoves[elemIndex];
            }
        }
    }
}
