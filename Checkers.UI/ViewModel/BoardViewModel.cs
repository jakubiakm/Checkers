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
        public ObservableCollection<Model.CanvasElement> BoardCanvasElements { get; } = new ObservableCollection<Model.CanvasElement>();

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

        public void RefreshBoard()
        {
            BoardCanvasElements.Clear();
            int skipSize = 700 / Game.Board.Size;
            int index = 0;
            for (int i = 0; i != Game.Board.Size; i++)
            {
                for (int j = 0; j != Game.Board.Size; j++)
                {
                    BoardCanvasElements.Add(new Model.CanvasElement
                    {
                        Row = skipSize * j,
                        Column = skipSize * i,
                        Geometry = new RectangleGeometry { Rect = new System.Windows.Rect(0, 0, skipSize, skipSize) },
                        Fill = index++ % 2 == 1 ? Brushes.CadetBlue : Brushes.AntiqueWhite
                    });
                }
                index++;
            }
            if (Game.LastMove != null)
            {
                BoardCanvasElements.Add(new Model.CanvasElement
                {
                    Row = skipSize * (Game.Board.Size - 1 - Game.LastMove.OldPiece.Row),
                    Column = skipSize * Game.LastMove.OldPiece.Column,
                    Geometry = new RectangleGeometry { Rect = new System.Windows.Rect(0, 0, skipSize, skipSize) },
                    Fill = Brushes.GreenYellow
                });
                BoardCanvasElements.Add(new Model.CanvasElement
                {
                    Row = skipSize * (Game.Board.Size - 1 - Game.LastMove.NewPiece.Row),
                    Column = skipSize * Game.LastMove.NewPiece.Column,
                    Geometry = new RectangleGeometry { Rect = new System.Windows.Rect(0, 0, skipSize, skipSize) },
                    Fill = Brushes.GreenYellow
                });
                foreach (var piece in Game.LastMove?.BeatedPieces ?? new List<Logic.GameObjects.Piece>())
                {
                    BoardCanvasElements.Add(new Model.CanvasElement
                    {
                        Row = skipSize * (Game.Board.Size - 1 - piece.Row),
                        Column = skipSize * piece.Column,
                        Geometry = new RectangleGeometry { Rect = new System.Windows.Rect(0, 0, skipSize, skipSize) },
                        Fill = Brushes.Crimson
                    });
                }
            }
            foreach (var elem in Game.Board.PiecesOnBoard)
            {
                BoardCanvasElements.Add(new Model.CanvasElement
                {
                    Row = skipSize * (Game.Board.Size - 1 - elem.Row) + skipSize / 2,
                    Column = skipSize * elem.Column + skipSize / 2,
                    Geometry = new EllipseGeometry { RadiusX = skipSize / 3, RadiusY = skipSize / 3 },
                    Fill = elem.Color == PieceColor.Black ? Brushes.Black : Brushes.White
                });
                if (elem.IsKing)
                {
                    BoardCanvasElements.Add(new Model.CanvasElement
                    {
                        Row = skipSize * (Game.Board.Size - 1 - elem.Row) + skipSize / 2,
                        Column = skipSize * elem.Column + skipSize / 2,
                        Geometry = new EllipseGeometry { RadiusX = skipSize / 4, RadiusY = skipSize / 4 },
                        Stroke = elem.Color == PieceColor.Black ? Brushes.White : Brushes.Black
                    });
                }
            }
        }
    }
}
