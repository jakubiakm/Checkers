using Checkers.Logic.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.GameObjects
{
    public class HistoryBoard
    {
        public DateTime StartTime { get; set; }

        public DateTime EndTime { get; set; }

        public CheckersBoard Board { get; set; }

        public PieceColor PieceColor { get; set; }

        public HistoryBoard(DateTime startTime, DateTime endTime, CheckersBoard board, PieceColor pieceColor)
        {
            StartTime = startTime;
            EndTime = endTime;
            Board = board;
            PieceColor = pieceColor;
        }
    }
}
