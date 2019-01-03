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
    public class RandomEngine : IEngine
    {
        public EngineKind Kind
        {
            get
            {
                return EngineKind.Random;
            }
        }

        public PieceColor Color { get; set; }

        public int? Seed { get; private set; }

        private Random randomGenerator;

        public RandomEngine(PieceColor color, int? seed)
        {
            Seed = seed;
            if (Seed != null)
                randomGenerator = new Random((int)Seed);
            else
                randomGenerator = new Random();
            Color = color;
        }

        public void Reset()
        {
            if (Seed != null)
                randomGenerator = new Random((int)Seed);
            else
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
