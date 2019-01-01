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

        public int Size { get; set; }

        public int Row
        {
            get
            {
                return _row;
            }
            set
            {
                _row = value;
                _position = ToPosition(_row, _column, Size);
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
                _position = ToPosition(_row, _column, Size);
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
                _row = ToRow(value, Size);
                _column = ToColumn(value, Size);
            }
        }

        public bool IsKing { get; set; }

        public PieceColor Color { get; internal set; }

        public Piece(int row, int column, PieceColor color, int boardSize, bool isKing)
        {
            Size = boardSize;
            Row = row;
            Column = column;
            Color = color;
            IsKing = isKing;
        }

        public Piece(int position, PieceColor color, int boardSize, bool isKing)
        {
            Size = boardSize;
            Position = position;
            Color = color;
            IsKing = isKing;
        }

        public override string ToString()
        {
            return $"[({Row},{Column}),King:{IsKing},{Color}]";
        }

        public static int ToPosition(int row, int column, int size)
        {
            if ((column % 2 == 0 && row % 2 == 1) || (row % 2 == 0 && column % 2 == 1))
                return -1;
            else
                return size / 2 * (size - row - 1) + ((row % 2 == 0) ? 1 : 0) + (column + 1) / 2;
        }

        public static int ToRow(int position, int size)
        {
            int div = size / 2;
            return size - 1 - (position - 1) / div;
        }

        public static int ToColumn(int position, int size)
        {
            int div = size / 2;
            return 2 * ((position - 1) % div) + (ToRow(position, size) % 2 == 1 ? 1 : 0);
        }
    }
}