using Checkers.Logic.Enums;
using Checkers.Logic.GameObjects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.Exceptions
{
    public class NoAvailablePiecesException : Exception
    {
        public NoAvailablePiecesException(PieceColor color, Move move)
        {
            Color = color;
            LastMove = move;
        }
        public PieceColor Color { get; set; }

        public Move LastMove { get; set; }
    }
}
