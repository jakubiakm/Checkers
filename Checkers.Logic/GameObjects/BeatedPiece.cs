﻿using Checkers.Logic.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.GameObjects
{
    [Serializable]
    public class BeatedPiece : Piece
    {
        public BeatedPiece(int row, int column, PieceColor color, bool isKing, int beatRow, int beatColumn) : base(row, column, color, 10, isKing)
        {
            Row = row;
            Column = column;
            Color = color;
            IsKing = isKing;
            BeatPieceRow = beatRow;
            BeatPieceColumn = beatColumn;
        }
        public int BeatPieceRow { get; set; }

        public int BeatPieceColumn { get; set; }
    }
}
