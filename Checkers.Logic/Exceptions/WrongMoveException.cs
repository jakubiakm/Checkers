using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.Exceptions
{
    public class WrongMoveException : Exception
    {
        public WrongMoveException(int min, string possibleMoves)
        {
            MinimumBeatedPieces = min;
            PossibleMoves = possibleMoves;
        }

        public int MinimumBeatedPieces { get; set; }

        public string PossibleMoves { get; set; }
    }
}
