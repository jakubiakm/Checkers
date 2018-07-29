using Checkers.Logic.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.Exceptions
{
    public class NotAvailableMoveException : Exception
    {
        public NotAvailableMoveException(PieceColor color) => Color = color;

        public PieceColor Color { get; set; }
    }
}
