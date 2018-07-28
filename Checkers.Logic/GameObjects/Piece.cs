using Checkers.Logic.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.GameObjects
{
    public class Piece
    {
        public int Row { get; set; }

        public int Column { get; set; }

        public bool IsKing { get; set; }
        
        public PieceColor Color { get; private set; }

        public Piece(int row, int column, PieceColor color)
        {
            Row = row;
            Column = column;
            Color = color;
            IsKing = false;
        }
    }
}
