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
    public class MctsEngine : IEngine
    {
        public EngineKind Kind
        {
            get
            {
                return EngineKind.Mcts;
            }
        }

        public PieceColor Color { get; set; }

        public int? Seed { get; private set; }

        public double UctParameter { get; private set; }

        public int NumberOfIterations { get; private set; }

        private Random randomGenerator;

        public MctsEngine(PieceColor color, int? seed, double uctParameter, int numberOfIterations)
        {
            Seed = seed;
            if (Seed != null)
                randomGenerator = new Random((int)Seed);
            else
                randomGenerator = new Random();
            Color = color;
            UctParameter = uctParameter;
            NumberOfIterations = numberOfIterations;
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
                MctsTree tree = new MctsTree(NumberOfIterations, UctParameter, randomGenerator, currentBoard, Color);
                int elemIndex = tree.ChooseBestMove(variant);
                return allPossibleMoves[elemIndex];
            }
        }
    }
}
