using Checkers.Logic.Enums;
using Checkers.Logic.GameObjects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.Engines
{
    public class RandomEngine : IEngine
    {
        public PieceColor Color { get; set; }

        public RandomEngine(PieceColor color)
        {
            Color = color;
        }

        public Move MakeMove(CheckersBoard currentBoard)
        {
            Random random = new Random();
            List<Move> allPossibleMoves = currentBoard.GetAllPossibleMoves(Color);
            int count = allPossibleMoves.Count;
            if (count == 0)
                throw new NotImplementedException();
            int elemIndex = random.Next(count - 1);       
            return allPossibleMoves[elemIndex];
        }
    }
}
