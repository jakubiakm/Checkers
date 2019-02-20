using Checkers.Logic.Enums;
using Checkers.Logic.GameObjects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.AlgorithmObjects
{
    public class MctsNode
    {
        public MctsNode Parent { get; set; }

        public List<MctsNode> Children { get; set; }

        public double NumberOfWins { get; set; }

        public int NumberOfSimulations { get; set; }

        public CheckersBoard Board { get; set; }

        public PieceColor Color { get; set; }

        public MctsNode(PieceColor color, CheckersBoard board)
        {
            Color = color;
            Board = board;
            NumberOfSimulations = 0;
            NumberOfWins = 0;
        }
        public MctsNode(PieceColor color, CheckersBoard board, MctsNode parent)
        {
            Color = color;
            Board = board;
            Parent = parent;
            NumberOfSimulations = 0;
            NumberOfWins = 0;
        }
    }
}
