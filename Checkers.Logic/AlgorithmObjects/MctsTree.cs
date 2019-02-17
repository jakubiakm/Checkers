using Checkers.Logic.Enums;
using Checkers.Logic.GameObjects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.AlgorithmObjects
{
    public class MctsTree
    {
        public MctsNode Root { get; set; }

        public int NumberOfIterations { get; set; }

        public double UctParameter { get; set; }

        public Random RandomGenerator { get; set; }

        public int NumberOfTotalSimulations { get; set; }

        public MctsTree(int numberOfIterations, double uctParameter, Random generator, CheckersBoard board, PieceColor color)
        {
            NumberOfTotalSimulations = 0;
            NumberOfIterations = numberOfIterations;
            UctParameter = uctParameter;
            RandomGenerator = generator;
            Root = new MctsNode(color == PieceColor.White ? PieceColor.Black : PieceColor.White, board);
        }

        public int ChooseBestMove(GameVariant variant)
        {
            for (int it = 0; it < NumberOfIterations; it++)
            {
                int result = 0;
                var leaf = Select();
                if (leaf.NumberOfSimulations == 0)
                {
                    result = Rollout(leaf);
                }
                else
                {
                    Expand(leaf);
                    if (leaf.Children != null && leaf.Children.Count > 0)
                    {
                        leaf = leaf.Children[0];
                        result = Rollout(leaf);
                    }
                    else
                    {
                        //w przypadku gdy po fazie "Expand" nie ma liści, to znaczy, że gra została już zakończona
                        //i wygrywa gracz, który ma jeszcze możliwe ruchy
                        result = leaf.Color == PieceColor.Black ? 1 : -1;
                    }
                }
                Backpropagate(result, leaf);
                NumberOfTotalSimulations++;
            }
            int index = 0;
            int maxSimulations = 0;
            for (int i = 0; i != Root.Children.Count; i++)
            {
                if (maxSimulations > Root.Children[i].NumberOfSimulations)
                {
                    maxSimulations = Root.Children[i].NumberOfSimulations;
                    index = i;
                }
            }
            return index;
        }

        /// <summary>
        /// Wybieramy liść w drzewie za pomocą doboru wartości UCT
        /// </summary>
        /// <returns></returns>
        private MctsNode Select()
        {
            var node = Root;
            while (node.Children != null && node.Children.Count > 0)
            {
                int index = 0;
                double maxUct = double.MinValue;
                double currentUct = 0;

                for (int i = 0; i < node.Children.Count; i++)
                {
                    // gdy jakaś gałąż nie była w ogóle symulowana, to obliczenia przerywamy i wchodzimy do tej gałęzi 
                    if (node.Children[i].NumberOfSimulations == 0)
                    {
                        index = i;
                        break;
                    }
                    currentUct = node.Children[i].NumberOfWins / node.Children[i].NumberOfSimulations + UctParameter * Math.Sqrt(Math.Log(NumberOfTotalSimulations) / node.Children[i].NumberOfSimulations);
                    if (currentUct > maxUct)
                    {
                        index = i;
                        maxUct = currentUct;
                    }
                }
                node = node.Children[index];
            }
            return node;
        }

        /// <summary>
        /// Dodajemy do liścia nowe gałęzie odpowiadające kolejnym ruchom z danego stanu
        /// </summary>
        /// <param name="node"></param>
        private void Expand(MctsNode node)
        {
            var moves = node.Board.GetAllPossibleMoves(node.Color);
            if (moves.Count > 0)
            {
                var children = new List<MctsNode>();
                foreach (var move in moves)
                {
                    var board = node.Board.GetBoardAfterMove(move);
                    var color = node.Color == PieceColor.White ? PieceColor.Black : PieceColor.White;
                    children.Add(new MctsNode(color, board, node));
                }
                node.Children = children;
            }
        }

        /// <summary>
        /// Wykonujemy symulacje
        /// </summary>
        /// <returns></returns>
        private int Rollout(MctsNode node)
        {
            int result = 0;
            CheckersBoard board = node.Board;
            PieceColor color = node.Color;
            while (true)
            {
                var moves = board.GetAllPossibleMoves(color);
                if (moves.Count > 0)
                {
                    board = board.GetBoardAfterMove(moves[RandomGenerator.Next(0, moves.Count - 1)]);
                }
                else
                {
                    result = color == PieceColor.Black ? 1 : -1;
                    break;
                }
                color = color == PieceColor.Black ? PieceColor.White : PieceColor.Black;
            }
            return result;
        }

        /// <summary>
        /// Wrzucamy wyniki wyżej
        /// </summary>
        private void Backpropagate(int result, MctsNode node)
        {
            while (node != null)
            {
                node.NumberOfSimulations++;
                if (result == 1 && node.Color == PieceColor.White)
                    node.NumberOfWins++;
                if (result == -1 && node.Color == PieceColor.Black)
                    node.NumberOfWins++;
                node = node.Parent;
            }
        }
    }
}
