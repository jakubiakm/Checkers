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
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Shapes;

namespace Checkers.UI.ViewModel
{
    public class BoardViewModel
    {
        public BoardViewModel(MainWindow window)
        {
            Window = window;
        }

        public bool WhiteIsHumnan { get; set; }

        public bool BlackIsHuman { get; set; }

        public PieceColor CurrentPlayer { get; set; }

        public MainWindow Window { get; set; }

        public ObservableCollection<CanvasElement> BoardCanvasElements { get; } = new ObservableCollection<CanvasElement>();

        public Game Game { get; private set; }

        public bool AnimationCompleted { get; private set; } = true;

        public int MoveAnimationTime { get; set; }

        private Move lastMove = null;

        double skipSize = 0;

        public void StartNewGame(
            int boardSize,
            int whiteCountSize,
            int blackCountSize,
            GameVariant gameVariant,
            EngineKind whiteEngineKind,
            EngineKind blackEngineKind,
            int moveAnimationTime)
        {

            CurrentPlayer = PieceColor.White;
            MoveAnimationTime = moveAnimationTime;
            Game = new Game(
                ConvertEngineKindEnumToEngine(whiteEngineKind, PieceColor.White),
                ConvertEngineKindEnumToEngine(blackEngineKind, PieceColor.Black),
                boardSize,
                whiteCountSize,
                blackCountSize,
                gameVariant);
            WhiteIsHumnan = Game.WhitePlayerEngine is HumanEngine;
            BlackIsHuman = Game.BlackPlayerEngine is HumanEngine;
            skipSize = 700 / Game.Board.Size;
            DrawCurrentBoard();
        }

        public IEngine ConvertEngineKindEnumToEngine(EngineKind kind, PieceColor color)
        {
            switch (kind)
            {
                case EngineKind.Human:
                    return new HumanEngine(color);
                case EngineKind.Random:
                    return new RandomEngine(color);
                default:
                    throw new ArgumentException("Nierozpoznany typ silnika");
            }
        }

        public Move NextMove(List<Path> humanMoves = null)
        {
            if (AnimationCompleted)
            {
                if (humanMoves != null && CurrentPlayer == PieceColor.White && WhiteIsHumnan)
                {
                    ((HumanEngine)Game.WhitePlayerEngine).HumanMove = ConvertPathListToPieceList(humanMoves);
                }

                if (humanMoves != null && CurrentPlayer == PieceColor.Black && BlackIsHuman)
                {
                    ((HumanEngine)Game.BlackPlayerEngine).HumanMove = ConvertPathListToPieceList(humanMoves);
                }

                Move move;
                if (CurrentPlayer == PieceColor.White)
                    move = Game.MakeMove(PieceColor.White);
                else
                    move = Game.MakeMove(PieceColor.Black);

                CurrentPlayer = CurrentPlayer == PieceColor.Black ? PieceColor.White : PieceColor.Black;

                return move;
            }
            return null;
        }

        private List<Piece> ConvertPathListToPieceList(List<Path> paths)
        {
            List<Piece> ret = new List<Piece>();
            foreach (var path in paths)
            {
                var elem = BoardCanvasElements.SingleOrDefault(e => e.Geometry == path.Data);
                ret.Add(new Piece(elem.X, elem.Y, PieceColor.White, Game.Board.Size, false));
            }
            return ret;
        }

        public void DrawCurrentBoard()
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
                        Fill = index++ % 2 == 1 ? Brushes.Peru : Brushes.AntiqueWhite,
                        X = Game.Board.Size - 1 - j,
                        Y = i
                    });
                }
                index++;
            }
            if (Game.Board.LastMove != null)
            {
                foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is RectangleGeometry && e.X == Game.Board.LastMove.OldPiece.Row && e.Y == Game.Board.LastMove.OldPiece.Column).ToList())
                {
                    elem.Fill = Brushes.DarkOrange;
                }
                foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is RectangleGeometry && e.X == Game.Board.LastMove.NewPiece.Row && e.Y == Game.Board.LastMove.NewPiece.Column).ToList())
                {
                    elem.Fill = Brushes.DarkOrange;
                }
                foreach (var piece in Game.Board.LastMove?.BeatedPieces ?? new List<BeatedPiece>())
                {
                    foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is RectangleGeometry && e.X == piece.Row && e.Y == piece.Column).ToList())
                    {
                        elem.Fill = Brushes.Crimson;
                    }
                }
            }
            foreach (var elem in Game.Board.PiecesOnBoard)
            {
                BoardCanvasElements.Add(new Model.CanvasElement
                {
                    Row = skipSize * (Game.Board.Size - 1 - elem.Row) + skipSize / 2,
                    Column = skipSize * elem.Column + skipSize / 2,
                    Geometry = new EllipseGeometry { RadiusX = skipSize / 3, RadiusY = skipSize / 3 },
                    Fill = elem.Color == PieceColor.Black ? Brushes.Black : Brushes.White,
                    X = elem.Row,
                    Y = elem.Column,
                    Color = elem.Color
                });
                if (elem.IsKing)
                {
                    BoardCanvasElements.Add(new Model.CanvasElement
                    {
                        Row = skipSize * (Game.Board.Size - 1 - elem.Row) + skipSize / 2,
                        Column = skipSize * elem.Column + skipSize / 2,
                        Geometry = new EllipseGeometry { RadiusX = skipSize / 4, RadiusY = skipSize / 4 },
                        Stroke = elem.Color == PieceColor.Black ? Brushes.White : Brushes.Black,
                        Thickness = 2,
                        X = elem.Row,
                        Y = elem.Column,
                        Color = elem.Color
                    });
                }
            }
        }

        public void DrawHistoryBoard(int moveNumber)
        {
            if (moveNumber == Game.History.Count)
            {
                DrawCurrentBoard();
                return;
            }
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
                        Fill = index++ % 2 == 1 ? Brushes.Peru : Brushes.AntiqueWhite,
                        X = Game.Board.Size - 1 - j,
                        Y = i
                    });
                }
                index++;
            }
            foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is RectangleGeometry && e.X == Game.History[moveNumber].LastMove.OldPiece.Row && e.Y == Game.History[moveNumber].LastMove.OldPiece.Column).ToList())
            {
                elem.Fill = Brushes.DarkOrange;
            }
            foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is RectangleGeometry && e.X == Game.History[moveNumber].LastMove.NewPiece.Row && e.Y == Game.History[moveNumber].LastMove.NewPiece.Column).ToList())
            {
                elem.Fill = Brushes.DarkOrange;
            }
            foreach (var piece in Game.History[moveNumber].LastMove?.BeatedPieces ?? new List<BeatedPiece>())
            {
                foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is RectangleGeometry && e.X == piece.Row && e.Y == piece.Column).ToList())
                {
                    elem.Fill = Brushes.Crimson;
                }
            }
            foreach (var elem in Game.History[moveNumber].PiecesOnBoard)
            {
                BoardCanvasElements.Add(new Model.CanvasElement
                {
                    Row = skipSize * (Game.Board.Size - 1 - elem.Row) + skipSize / 2,
                    Column = skipSize * elem.Column + skipSize / 2,
                    Geometry = new EllipseGeometry { RadiusX = skipSize / 3, RadiusY = skipSize / 3 },
                    Fill = elem.Color == PieceColor.Black ? Brushes.Black : Brushes.White,
                    X = elem.Row,
                    Y = elem.Column
                });
                if (elem.IsKing)
                {
                    BoardCanvasElements.Add(new Model.CanvasElement
                    {
                        Row = skipSize * (Game.Board.Size - 1 - elem.Row) + skipSize / 2,
                        Column = skipSize * elem.Column + skipSize / 2,
                        Geometry = new EllipseGeometry { RadiusX = skipSize / 4, RadiusY = skipSize / 4 },
                        Stroke = elem.Color == PieceColor.Black ? Brushes.White : Brushes.Black,
                        Thickness = 2,
                        X = elem.Row,
                        Y = elem.Column
                    });
                }
            }
        }

        public void DrawNextMove(Move move)
        {
            RemoveLastMoveTargets(move);
            DrawMoveTargets(move);
            AnimateMovement(move);

            lastMove = move;
        }

        private void RemoveLastMoveTargets(Move move)
        {
            if (lastMove != null)
            {
                foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is RectangleGeometry && e.X == lastMove.OldPiece.Row && e.Y == lastMove.OldPiece.Column).ToList())
                {
                    elem.Fill = Brushes.Peru;
                }
                foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is RectangleGeometry && e.X == lastMove.NewPiece.Row && e.Y == lastMove.NewPiece.Column).ToList())
                {
                    elem.Fill = Brushes.Peru;
                }
                foreach (var piece in lastMove?.BeatedPieces ?? new List<BeatedPiece>())
                {
                    foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is RectangleGeometry && e.X == piece.Row && e.Y == piece.Column).ToList())
                    {
                        elem.Fill = Brushes.Peru;
                    }
                }
            }
        }

        private void DrawMoveTargets(Move move)
        {
            foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is RectangleGeometry && e.X == move.OldPiece.Row && e.Y == move.OldPiece.Column).ToList())
            {
                elem.Fill = Brushes.DarkOrange;
            }
            foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is RectangleGeometry && e.X == move.NewPiece.Row && e.Y == move.NewPiece.Column).ToList())
            {
                elem.Fill = Brushes.DarkOrange;
            }
            foreach (var piece in Game.Board.LastMove?.BeatedPieces ?? new List<BeatedPiece>())
            {
                foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is RectangleGeometry && e.X == piece.Row && e.Y == piece.Column).ToList())
                {
                    elem.Fill = Brushes.Crimson;
                }
            }
        }

        private void DrawMoveMovement(Move move, CanvasElement canvasElem)
        {
            foreach (var piece in move.BeatedPieces ?? new List<BeatedPiece>())
            {
                foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is EllipseGeometry && e.X == piece.Row && e.Y == piece.Column).ToList())
                {
                    BoardCanvasElements.Remove(elem);
                }
            }

            canvasElem.Row = skipSize * (Game.Board.Size - 1 - move.NewPiece.Row) + skipSize / 2;
            canvasElem.Column = skipSize * move.NewPiece.Column + skipSize / 2;
            canvasElem.X = move.NewPiece.Row;
            canvasElem.Y = move.NewPiece.Column;

            if (move.NewPiece.IsKing && !move.OldPiece.IsKing)
            {
                BoardCanvasElements.Add(new Model.CanvasElement
                {
                    Row = skipSize * (Game.Board.Size - 1 - move.NewPiece.Row) + skipSize / 2,
                    Column = skipSize * move.NewPiece.Column + skipSize / 2,
                    Geometry = new EllipseGeometry { RadiusX = skipSize / 4, RadiusY = skipSize / 4 },
                    Thickness = 2,
                    Stroke = move.NewPiece.Color == PieceColor.Black ? Brushes.White : Brushes.Black,
                    X = move.NewPiece.Row,
                    Y = move.NewPiece.Column
                });
            }
        }

        private void AnimateMovement(Move move)
        {
            foreach (var elem in BoardCanvasElements.Where(e => e.Geometry is EllipseGeometry && e.X == move.OldPiece.Row && e.Y == move.OldPiece.Column).ToList())
            {
                var x = elem.Geometry;
                if (Window.FindName("MyAnimatedEllipseGeometry") != null)
                    Window.UnregisterName("MyAnimatedEllipseGeometry");
                Window.RegisterName("MyAnimatedEllipseGeometry", x);

                PathGeometry animationPath = new PathGeometry();
                Point position = new Point(move.OldPiece.Column, move.OldPiece.Row);
                PathFigure pFigure = new PathFigure();
                pFigure.StartPoint = new Point(0, 0);
                PolyLineSegment lineSegment = new PolyLineSegment();
                double xDiff, yDiff;
                foreach (var beatPositionPiece in move.BeatedPieces?.Skip(1) ?? new List<BeatedPiece>())
                {
                    xDiff = ((skipSize * beatPositionPiece.BeatPieceColumn + skipSize / 2) - (skipSize * position.X + skipSize / 2));
                    yDiff = (skipSize * (Game.Board.Size - 1 - beatPositionPiece.BeatPieceRow) + skipSize / 2) - (skipSize * (Game.Board.Size - 1 - position.Y) + skipSize / 2);
                    if (lineSegment.Points.Count > 0)
                        lineSegment.Points.Add(new Point(lineSegment.Points.Last().X + xDiff, lineSegment.Points.Last().Y + yDiff));
                    else
                        lineSegment.Points.Add(new Point(xDiff, yDiff));
                    position = new Point(beatPositionPiece.BeatPieceColumn, beatPositionPiece.BeatPieceRow);
                }
                xDiff = (skipSize * move.NewPiece.Column + skipSize / 2) - (skipSize * position.X + skipSize / 2);
                yDiff = (skipSize * (Game.Board.Size - 1 - move.NewPiece.Row) + skipSize / 2) - (skipSize * (Game.Board.Size - 1 - position.Y) + skipSize / 2);
                if (lineSegment.Points.Count > 0)
                    lineSegment.Points.Add(new Point(lineSegment.Points.Last().X + xDiff, lineSegment.Points.Last().Y + yDiff));
                else
                    lineSegment.Points.Add(new Point(xDiff, yDiff));
                pFigure.Segments.Add(lineSegment);
                animationPath.Figures.Add(pFigure);
                animationPath.Freeze();


                PointAnimationUsingPath myPointAnimation = new PointAnimationUsingPath();
                myPointAnimation.Duration = TimeSpan.FromSeconds((double)MoveAnimationTime / 100 * ((move.BeatedPieces?.Count()) ?? 1));
                myPointAnimation.FillBehavior = FillBehavior.Stop;
                myPointAnimation.PathGeometry = animationPath;

                AnimationCompleted = false;
                myPointAnimation.Completed += delegate
                {
                    DrawMoveMovement(move, elem);
                    AnimationCompleted = true;
                };

                // Set the animation to target the Center property
                // of the object named "MyAnimatedEllipseGeometry."
                Storyboard.SetTargetName(myPointAnimation, "MyAnimatedEllipseGeometry");
                Storyboard.SetTargetProperty(
                    myPointAnimation, new PropertyPath(EllipseGeometry.CenterProperty));

                // Create a storyboard to apply the animation.
                Storyboard ellipseStoryboard = new Storyboard();
                ellipseStoryboard.Children.Add(myPointAnimation);
                ellipseStoryboard.Begin(Window);
                elem.Geometry = x;
            }
        }

        private void MyPointAnimation_Completed(object sender, EventArgs e)
        {
            throw new NotImplementedException();
        }
    }
}
