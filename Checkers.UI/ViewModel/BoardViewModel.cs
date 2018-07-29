using Checkers.Logic.Engines;
using Checkers.Logic.Enums;
using Checkers.Logic.GameObjects;
using Checkers.UI.Model;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;

namespace Checkers.UI.ViewModel
{
    public class BoardViewModel
    {
        public ObservableCollection<Model.Piece> Pieces { get; } = new ObservableCollection<Model.Piece>();

        public Game Game { get; private set; }

        int turn = 0;

        public void StartNewGame()
        {
            Game = new Game(new RandomEngine(PieceColor.White), new RandomEngine(PieceColor.Black));
            RefreshBoard();
        }

        public void NextMove()
        {
            if (turn++ % 2 == 0)
                Game.MakeMove(PieceColor.White);
            else
                Game.MakeMove(PieceColor.Black);
            RefreshBoard();
        }

        private void RefreshBoard()
        {
            Pieces.Clear();
            int skipSize = 700 / Game.Board.Size;
            int index = 0;
            for (int i = 0; i != Game.Board.Size; i++)
            {
                for (int j = 0; j != Game.Board.Size; j++)
                {
                    Pieces.Add(new Model.Piece
                    {
                        Row = skipSize * j,
                        Column = skipSize * i,
                        Geometry = new RectangleGeometry { Rect = new System.Windows.Rect(0, 0, skipSize, skipSize) },
                        Fill = index++ % 2 == 1 ? Brushes.CadetBlue : Brushes.AntiqueWhite
                    });
                }
                index++;
            }
            foreach (var elem in Game.Board.PiecesOnBoard)
            {
                Pieces.Add(new Model.Piece
                {
                    Row = skipSize * (Game.Board.Size - 1 - elem.Row) + skipSize / 2,
                    Column = skipSize * elem.Column + skipSize / 2,
                    Geometry = new EllipseGeometry { RadiusX = skipSize / 3, RadiusY = skipSize / 3 },
                    Fill = elem.Color == PieceColor.Black ? Brushes.Black : Brushes.White
                });
            }
        }
    }
}
