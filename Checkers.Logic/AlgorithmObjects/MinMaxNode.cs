using Checkers.Logic.Enums;
using Checkers.Logic.GameObjects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.AlgorithmObjects
{
    public class MinMaxNode
    {
        public MinMaxNode Parent { get; set; }

        public List<MinMaxNode> Children { get; set; }

        public int DepthLevel { get; set; }
        
        public int CurrentScore { get; set; }

        public CheckersBoard Board { get; set; }

        public PieceColor Color { get; set; }

        public int Score
        {
            get
            {
                return GetScore();
            }
        }

        public MinMaxNode(CheckersBoard board, PieceColor color, int depthLevel)
        {
            Board = board;
            Color = color;
            DepthLevel = depthLevel;
        }

        private int GetScore()
        {
            int score = 0;
            foreach(var piece in Board.PiecesOnBoard)
            {
                //za każdy pionek odpowiedniego koloru dodajemy lub odejmujemy punkty
                //w przypadku królowych punkty są warte potrójną wartość
                if (piece.Color == PieceColor.White && piece.IsKing)
                    score += 3;
                if (piece.Color == PieceColor.White && !piece.IsKing)
                    score += 1;
                if (piece.Color == PieceColor.Black && piece.IsKing)
                    score -= 3;
                if (piece.Color == PieceColor.Black && !piece.IsKing)
                    score -= 1;
            }
            return score;
        }
    }
}
