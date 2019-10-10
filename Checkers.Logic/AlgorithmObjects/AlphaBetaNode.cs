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
            for (int ind = 1; ind != Board.BoardArray.Length; ind++)
            {
                //-2 - czarna dama
                //-1 - czarny pion
                // 0 - brak piona
                // 1 - biały pion
                // 2 - biała dama

                //za każdy pionek odpowiedniego koloru dodajemy lub odejmujemy 2 punkty
                //w przypadku królowych dostajemy lub tracimy 6 punktów
                if (Board.BoardArray[ind] == 2)
                    score += 6;
                if (Board.BoardArray[ind] == 1)
                    score += 2;
                if (Board.BoardArray[ind] == -2)
                    score -= 6;
                if (Board.BoardArray[ind] == -1)
                    score -= 2;

                //czarne pionki w pierwszym rzędzie są dodatkowo punktowane dla czarnych 
                if(Board.BoardArray[ind] < 0 && Piece.ToRow(ind, Board.Size) == Board.Size - 1)
                {
                    score -= 1;
                }

                //białe pionki ostatnim rzędzie planszy są dodatkowo punktowane dla białych
                if (Board.BoardArray[ind] > 0 && Piece.ToRow(ind, Board.Size) == 0)
                {
                    score += 1;
                }
            }

            //w przypadku antywarcabów odwracamy znak wyniku (dążymy do najgorszej pozycji)
            if (variant == GameVariant.Anticheckers)
                score = -score;
            return score;
        }
    }
}
