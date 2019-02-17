﻿using Checkers.Logic.Enums;
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

        public AlphaBetaTree(int depth)
        {
            Depth = depth;
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

        public int ChooseBestMove(GameVariant variant)
        {
            GetScore(variant, Root, int.MinValue, int.MaxValue);
            int index = Root.Children.FindIndex(n => n.CurrentScore == Root.CurrentScore);
            return index;
        }

        private int GetScore(GameVariant variant, AlphaBetaNode node, int alpha, int beta)
        {
            if (node.Children == null || node.Children.Count == 0)
            {
                node.CurrentScore = node.GetHeuristicScore(variant);
                return node.GetHeuristicScore(variant);
            }
            if (node.Color == PieceColor.Black)
            {
                foreach (var n in node.Children)
                {
                    beta = new List<int>() { beta, GetScore(variant, n, alpha, beta) }.Min();
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
                    alpha = new List<int>() { alpha, GetScore(variant, n, alpha, beta) }.Max();
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
