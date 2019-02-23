using Checkers.Logic.Enums;
using Checkers.Logic.GameObjects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.AlgorithmObjects
{
    public class AlphaBetaNode
    {
        public AlphaBetaNode Parent { get; set; }

        public List<AlphaBetaNode> Children { get; set; }

        public int DepthLevel { get; set; }

        public int CurrentScore { get; set; }

        public CheckersBoard Board { get; set; }

        public Move Move { get; set; }

        public PieceColor Color { get; set; }

        public AlphaBetaNode(CheckersBoard board, Move move, PieceColor color, int depthLevel)
        {
            Board = board;
            Color = color;
            DepthLevel = depthLevel;
            Move = move;
            if (Color == PieceColor.White)
                CurrentScore = int.MinValue;
            else
                CurrentScore = int.MaxValue;
        }

        public int GetHeuristicScore(GameVariant variant, bool isDraw)
        {
            int score = 0;
            if (isDraw)
                return 0;
            foreach (var piece in Board.BoardArray)
            {
                //za każdy pionek odpowiedniego koloru dodajemy lub odejmujemy punkty
                //w przypadku królowych punkty są warte potrójną wartość
                if (piece == 2)
                    score += 3;
                if (piece == 1)
                    score += 1;
                if (piece == -2)
                    score -= 3;
                if (piece == -1)
                    score -= 1;
            }

            //w przypadku antywarcabów odwracamy znak wyniku (dążymy do najgorszej pozycji)
            if (variant == GameVariant.Anticheckers)
                score = -score;
            return score;
        }
    }
}
