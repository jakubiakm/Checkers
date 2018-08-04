using Checkers.Logic.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.GameObjects
{
    [Serializable]
    public class Piece
    {
        private int _row;

        private int _column;

        private int _position;

        public int Row
        {
            get
            {
                return _row;
            }
            set
            {
                _row = value;
            }
        }

        public int Column
        {
            get
            {
                return _column;
            }
            set
            {
                _column = value;
            }
        }

        public int Position
        {
            get
            {
                return _position;
            }
            set
            {
                _position = value;
            }
        }

        public bool IsKing { get; set; }

        public PieceColor Color { get; internal set; }

        public Piece(int row, int column, PieceColor color, bool isKing)
        {
            Row = row;
            Column = column;
            Color = color;
            IsKing = isKing;
        }

        public Piece(int position, PieceColor color, bool isKing)
        {
            Position = position;
            Color = color;
            IsKing = isKing;
        }

        public override string ToString()
        {
            return $"[({Row},{Column}),King:{IsKing},{Color}]";
        }
    }
}
