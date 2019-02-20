using Checkers.Logic.Enums;
using Checkers.Logic.GameObjects;
using System.Threading;
using System.Collections.Generic;
using System.Linq;


namespace Checkers.Logic.AlgorithmObjects
{
    public class AlphaBetaTree
    {
        public AlphaBetaNode Root { get; set; }

        public int Depth { get; set; }

        public AlphaBetaTree(int depth, PieceColor color, CheckersBoard board)
        {
            Depth = depth;
            Root = new AlphaBetaNode(board, color, 0);
        }

        public int ChooseBestMove(GameVariant variant)
        {
            GetScore(variant, Root, int.MinValue, int.MaxValue, 0);
            int index = Root.Children.FindIndex(n => n.CurrentScore == Root.CurrentScore);
            return index;
        }

        private int GetScore(GameVariant variant, AlphaBetaNode node, int alpha, int beta, int depth)
        {
            List<Move> moves = node.Board.GetAllPossibleMoves(node.Color);
            List<AlphaBetaNode> childrens = new List<AlphaBetaNode>();
            foreach (var move in moves)
            {
                
                var board = node.Board.GetBoardAfterMove(move);
                var color = node.Color == PieceColor.Black ? PieceColor.White : PieceColor.Black;
                var child = new AlphaBetaNode(board, color, depth + 1);
                child.Parent = node;
                childrens.Add(child);
            }
            node.Children = childrens;
            if (node.Children == null || node.Children.Count == 0 || depth + 1 == Depth)
            {
                node.CurrentScore = node.GetHeuristicScore(variant);
                return node.GetHeuristicScore(variant);
            }
            if (node.Color == PieceColor.Black)
            {
                foreach (var n in node.Children)
                {
                    beta = new List<int>() { beta, GetScore(variant, n, alpha, beta, depth + 1) }.Min();
                    if (alpha >= beta)
                        break;
                }
                node.CurrentScore = beta;
                return beta;
            }
            else
            {
                foreach (var n in node.Children)
                {
                    alpha = new List<int>() { alpha, GetScore(variant, n, alpha, beta, depth + 1) }.Max();
                    if (alpha >= beta)
                        break;
                }
                node.CurrentScore = alpha;
                return alpha;
            }
        }

        //private int GetScoreMinMax(GameVariant variant, AlphaBetaNode node)
        //{
        //    if (node.Children == null || node.Children.Count == 0)
        //        return node.GetHeuristicScore(variant);
        //    return node.Color == PieceColor.White ? node.Children.Max(n => GetScoreMinMax(variant, n)) : node.Children.Min(n => GetScoreMinMax(variant, n));
        //}
    }
}
