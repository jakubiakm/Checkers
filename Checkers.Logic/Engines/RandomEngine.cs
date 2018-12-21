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
        public PieceColor Color { get; set; }

        public RandomEngine(PieceColor color)
        {
            Color = color;
        }

        [DllImport(@"D:\Users\syntaximus\Documents\GitHub\Checkers\x64\Debug\Checkers.Cuda.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall)]
        public static extern float GetExecutionTime();

        public Move MakeMove(CheckersBoard currentBoard)
        {
            Random random = new Random();
            List<Move> allPossibleMoves = currentBoard.GetAllPossibleMoves(Color);
            int count = allPossibleMoves.Count;
            if (count == 0)
                throw new NotAvailableMoveException(Color);
            int elemIndex = random.Next(count);
            var w = GetExecutionTime();
            return allPossibleMoves[elemIndex];
        }
    }
}
