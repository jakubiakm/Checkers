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

        public List<AlphaBetaNode> Leafs { get; set; }

        public int Depth { get; set; }

        public AlphaBetaTree(int depth)
        {
            Depth = depth;
            Leafs = new List<AlphaBetaNode>();
        }

        public void BuildTree(CheckersBoard board, PieceColor color)
        {
            Root = new AlphaBetaNode(board, color, 0);
            List<Move> moves = Root.Board.GetAllPossibleMoves(Root.Color);
            List<AlphaBetaNode> childrens = new List<AlphaBetaNode>();
            foreach (var move in moves)
            {
                var b = Root.Board.GetBoardAfterMove(move);
                var c = Root.Color == PieceColor.Black ? PieceColor.White : PieceColor.Black;
                var node = new AlphaBetaNode(b, c, 1);
                node.Parent = Root;
                childrens.Add(node);
            }
            Root.Children = childrens;
            Thread[] threads = new Thread[Root.Children.Count];
            for (int i = 0; i != Root.Children.Count; i++)
            {
                var n = Root.Children[i];
                threads[i] = new Thread(() =>
                {
                    GenerateNodes(n, 2);
                });
            }
            for (int i = 0; i != Root.Children.Count; i++)
            {
                threads[i].Start();
            }
            for (int i = 0; i != Root.Children.Count; i++)
            {
                threads[i].Join();
            }

        }

        public void GenerateNodes(AlphaBetaNode parent, int depth)
        {
            if (depth == Depth)
            {
                Leafs.Add(parent);
                return;
            }
            List<Move> moves = parent.Board.GetAllPossibleMoves(parent.Color);
            List<AlphaBetaNode> childrens = new List<AlphaBetaNode>();
            foreach (var move in moves)
            {
                var board = parent.Board.GetBoardAfterMove(move);
                var color = parent.Color == PieceColor.Black ? PieceColor.White : PieceColor.Black;
                var node = new AlphaBetaNode(board, color, depth + 1);
                node.Parent = parent;
                childrens.Add(node);
            }
            parent.Children = childrens;
            foreach (var node in parent.Children)
            {
                GenerateNodes(node, depth + 1);
            }
        }

        public int ChooseBestMove()
        {
            foreach (var node in Root.Children)
            {
                node.CurrentScore = GetScore(node, int.MinValue, int.MaxValue);
            }
            int index = 0, max = int.MinValue, min = int.MaxValue;
            for (int i = 0; i != Root.Children.Count; i++)
            {
                if (Root.Color == PieceColor.White)
                {
                    if (Root.Children[i].CurrentScore > max)
                    {
                        max = Root.Children[i].CurrentScore;
                        index = i;
                    }
                }
                else
                {
                    if (Root.Children[i].CurrentScore < min)
                    {
                        min = Root.Children[i].CurrentScore;
                        index = i;
                    }
                }
            }
            return index;
        }

        private int GetScore(AlphaBetaNode node, int alpha, int beta)
        {
            if (node.Children == null || node.Children.Count == 0)
                return node.Score;
            if (node.Color == PieceColor.Black)
            {
                foreach (var n in node.Children)
                {
                    beta = new List<int>() { beta, GetScore(n, alpha, beta) }.Min();
                    if (alpha >= beta)
                        break;
                }
                return beta;
            }
            else
            {
                foreach (var n in node.Children)
                {
                    alpha = new List<int>() { alpha, GetScore(n, alpha, beta) }.Max();
                    if (alpha >= beta)
                        break;
                }
                return alpha;
            }
        }
    }
}
