using Checkers.Logic.Enums;
using Checkers.Logic.GameObjects;
using System.Threading;
using System.Collections.Generic;
using System.Linq;


namespace Checkers.Logic.AlgorithmObjects
{
    public class MinMaxTree
    {
        public MinMaxNode Root { get; set; }

        public List<MinMaxNode> Leafs { get; set; }

        public int Depth { get; set; }

        public MinMaxTree(int depth)
        {
            Depth = depth;
            Leafs = new List<MinMaxNode>();
        }

        public void BuildTree(CheckersBoard board, PieceColor color)
        {
            Root = new MinMaxNode(board, color, 0);
            List<Move> moves = Root.Board.GetAllPossibleMoves(Root.Color);
            List<MinMaxNode> childrens = new List<MinMaxNode>();
            foreach (var move in moves)
            {
                var b = Root.Board.GetBoardAfterMove(move);
                var c = Root.Color == PieceColor.Black ? PieceColor.White : PieceColor.Black;
                var node = new MinMaxNode(b, c, 1);
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
            for(int i = 0; i != Root.Children.Count; i++)
            {
                threads[i].Start();
            }
            for (int i = 0; i != Root.Children.Count; i++)
            {
                threads[i].Join();
            }

        }

        public void GenerateNodes(MinMaxNode parent, int depth)
        {
            if (depth == Depth)
            {
                Leafs.Add(parent);
                return;
            }
            List<Move> moves = parent.Board.GetAllPossibleMoves(parent.Color);
            List<MinMaxNode> childrens = new List<MinMaxNode>();
            foreach (var move in moves)
            {
                var board = parent.Board.GetBoardAfterMove(move);
                var color = parent.Color == PieceColor.Black ? PieceColor.White : PieceColor.Black;
                var node = new MinMaxNode(board, color, depth + 1);
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
                node.CurrentScore = GetScore(node);
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

        private int GetScore(MinMaxNode node)
        {
            if (node.Children == null || node.Children.Count == 0)
                return node.Score;
            return node.Color == PieceColor.White ? node.Children.Max(n => GetScore(n)) : node.Children.Min(n => GetScore(n));
        }
    }
}
