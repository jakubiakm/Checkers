using Checkers.Logic.Engines;
using Checkers.Logic.Enums;
using Checkers.Logic.Exceptions;
using Checkers.Logic.GameObjects;
using Checkers.UI.ViewModel;
using Checkers.UI.Views;
using MahApps.Metro.Controls;
using MahApps.Metro.Controls.Dialogs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace Checkers.UI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : MetroWindow
    {
        public DispatcherTimer NotHumanMoveTimer { get; set; }

        private MetroWindow settingsWindow;

        public MainWindow()
        {
            InitializeComponent();
            BoardViewModelObject = new BoardViewModel(this);
            HistoryViewModelObject = new HistoryViewModel();
        }

        public void SetNotHumanMoveTimer()
        {
            NotHumanMoveTimer = new DispatcherTimer();
            NotHumanMoveTimer.Tick += Timer_Elapsed;
            NotHumanMoveTimer.Interval = new TimeSpan(0, 0, 0, 0, 1);
            NotHumanMoveTimer.Start();
        }

        private async void Timer_Elapsed(object sender, EventArgs e)
        {
            if (BoardViewModelObject.Game != null &&
                (BoardViewModelObject.CurrentPlayer == PieceColor.White && !BoardViewModelObject.WhiteIsHumnan) ||
                (BoardViewModelObject.CurrentPlayer == PieceColor.Black && !BoardViewModelObject.BlackIsHuman))
            {
                try
                {
                    var move = BoardViewModelObject.NextMove();
                    if (move != null)
                    {
                        BoardViewModelObject.DrawNextMove(move);
                        HistoryViewModelObject.AddHistoryItem(BoardViewModelObject.Game.Board.Size, move);
                    }
                }
                catch (NotAvailableMoveException exception)
                {
                    NotHumanMoveTimer.Stop();
                    await this.ShowMessageAsync("Remis", $"Gra zakończona remisem gracz {(exception.Color == Logic.Enums.PieceColor.Black ? "CZARNY" : "BIAŁY")} nie może już wykonywać ruchów.");
                    HistoryViewModelObject.History.Clear();
                    StartNewGame(
                        BoardViewModelObject.Game.Board.Size, 
                        BoardViewModelObject.Game.Board.NumberOfWhitePiecesAtBeggining,
                        BoardViewModelObject.Game.Board.NumberOfBlackPiecesAtBeggining,
                        BoardViewModelObject.Game.Variant,
                        BoardViewModelObject.Game.WhitePlayerEngine,
                        BoardViewModelObject.Game.BlackPlayerEngine,
                        BoardViewModelObject.MoveAnimationTime);
                    NotHumanMoveTimer.Start();
                }
                catch (NoAvailablePiecesException exception)
                {
                    NotHumanMoveTimer.Stop();
                    BoardViewModelObject.DrawNextMove(exception.LastMove);
                    await this.ShowMessageAsync("Koniec gry", $"Gra zakończony. Gracz {(exception.Color == Logic.Enums.PieceColor.Black ? "CZARNY" : "BIAŁY")} nie ma już pionków.");
                    HistoryViewModelObject.History.Clear();
                    StartNewGame(
                        BoardViewModelObject.Game.Board.Size,
                        BoardViewModelObject.Game.Board.NumberOfWhitePiecesAtBeggining,
                        BoardViewModelObject.Game.Board.NumberOfBlackPiecesAtBeggining,
                        BoardViewModelObject.Game.Variant,
                        BoardViewModelObject.Game.WhitePlayerEngine,
                        BoardViewModelObject.Game.BlackPlayerEngine,
                        BoardViewModelObject.MoveAnimationTime
                        );
                    NotHumanMoveTimer.Start();
                }
                catch (WrongMoveException exception)
                {
                    await this.ShowMessageAsync("Zły ruch", $"Gracz wykonał nielegalny ruch. Możliwa ilość pionków do bicia to {exception.MinimumBeatedPieces}.");

                }
            }
        }

        List<Path> HumanPlayerMove { get; set; } = new List<Path>();

        Canvas BoardCanvas { get; set; }

        ListView HistoryListView { get; set; }

        bool HistoryShowed { get; set; } = false;

        BoardViewModel BoardViewModelObject;

        HistoryViewModel HistoryViewModelObject;

        public async void StartNewGame(
            int boardSize, 
            int whiteCountSize, 
            int blackCountSize, 
            GameVariant gameVariant,
            IEngine whiteEngine,
            IEngine blackEngine,
            int moveAnimationTime)
        {
            try
            {
                BoardViewModelObject.StartNewGame(
                    boardSize, 
                    whiteCountSize, 
                    blackCountSize,
                    gameVariant,
                    whiteEngine,
                    blackEngine,
                    moveAnimationTime);
                HistoryViewModelObject.History.Clear();
                HistoryShowed = false;
            }
            catch (ArgumentException e)
            {
                await this.ShowMessageAsync("Błąd", $"{e.Message}");
            }
        }

        public void BoardViewControl_Loaded(object sender, RoutedEventArgs e)
        {
            StartNewGame(
                boardSize: 10,
                whiteCountSize: 20,
                blackCountSize: 20,
                gameVariant: GameVariant.Checkers,
                whiteEngine: new HumanEngine(PieceColor.White),
                blackEngine: new RandomEngine(PieceColor.Black, null),
                moveAnimationTime: 33);
            BoardViewControl.DataContext = BoardViewModelObject;
            BoardCanvas = UiHelper.FindChild<Canvas>(BoardViewControl, "BoardCanvas");
            BoardCanvas.MouseDown += BoardCanvas_MouseDown;
            SetNotHumanMoveTimer();
        }

        private async void BoardCanvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            IInputElement clickedElement = Mouse.DirectlyOver;
            if (clickedElement is Path && !HistoryShowed)
            {
                HistoryListView.SelectedIndex = -1;
                try
                {
                    if ((BoardViewModelObject.CurrentPlayer == PieceColor.White && BoardViewModelObject.WhiteIsHumnan)
                        || (BoardViewModelObject.CurrentPlayer == PieceColor.Black && BoardViewModelObject.BlackIsHuman))
                    {
                        var path = (Path)clickedElement;
                        if (path.Data is EllipseGeometry && path.StrokeThickness != 2)
                        {
                            if (HumanPlayerMove.Count != 0)
                            {
                                foreach (var p in HumanPlayerMove)
                                {
                                    if (p.Data is EllipseGeometry)
                                    {
                                        if (BoardViewModelObject.CurrentPlayer == PieceColor.White)
                                            p.Fill = Brushes.White;
                                        else
                                            p.Fill = Brushes.Black;
                                    }
                                    if (p.Data is RectangleGeometry)
                                    {
                                        p.Fill = Brushes.Peru;
                                    }
                                }
                                HumanPlayerMove.Clear();
                            }
                            if (GetPlayerColorFromPath(path) == BoardViewModelObject.CurrentPlayer)
                            {
                                HumanPlayerMove.Add(path);
                                path.Fill = Brushes.Gold;
                            }
                        }
                        if (path.Data is RectangleGeometry)
                        {
                            if (HumanPlayerMove.Count == 0)
                                return;
                            if (!(HumanPlayerMove.First().Data is EllipseGeometry))
                            {
                                foreach (var p in HumanPlayerMove)
                                {
                                    if (p.Data is EllipseGeometry)
                                    {
                                        if (BoardViewModelObject.CurrentPlayer == PieceColor.White)
                                            p.Fill = Brushes.White;
                                        else
                                            p.Fill = Brushes.Black;
                                    }
                                    if (p.Data is RectangleGeometry)
                                    {
                                        p.Fill = Brushes.Peru;
                                    }
                                }
                                HumanPlayerMove.Clear();
                                return;
                            }
                            if (path.Fill == Brushes.Gold && e.LeftButton == MouseButtonState.Pressed)
                            {

                                try
                                {
                                    foreach (var p in HumanPlayerMove)
                                    {
                                        if (p.Data is EllipseGeometry)
                                        {
                                            if (BoardViewModelObject.CurrentPlayer == PieceColor.White)
                                                p.Fill = Brushes.White;
                                            else
                                                p.Fill = Brushes.Black;
                                        }
                                        if (p.Data is RectangleGeometry)
                                        {
                                            p.Fill = Brushes.Peru;
                                        }
                                    }
                                    var move = BoardViewModelObject.NextMove(HumanPlayerMove);
                                    if (move != null)
                                    {
                                        BoardViewModelObject.DrawNextMove(move);
                                        HistoryViewModelObject.AddHistoryItem(BoardViewModelObject.Game.Board.Size, move);
                                    }
                                    else
                                    {

                                    }

                                }
                                catch (WrongMoveException exception)
                                {
                                    await this.ShowMessageAsync("Zły ruch", $"Gracz wykonał nielegalny ruch. Możliwa ilość pionków do bicia to {exception.MinimumBeatedPieces}. Możliwości są następujące: {exception.PossibleMoves}.");

                                }
                                finally
                                {
                                    HumanPlayerMove.Clear();
                                }
                            }
                            else if (path.Fill != Brushes.AntiqueWhite)
                            {
                                HumanPlayerMove.Add(path);
                                path.Fill = Brushes.Gold;
                            }
                        }
                    }
                }
                catch (NotAvailableMoveException exception)
                {
                    await this.ShowMessageAsync("Remis", $"Gra zakończona remisem gracz {(exception.Color == Logic.Enums.PieceColor.Black ? "CZARNY" : "BIAŁY")} nie może już wykonywać ruchów.");
                    StartNewGame(
                        BoardViewModelObject.Game.Board.Size,
                        BoardViewModelObject.Game.Board.NumberOfWhitePiecesAtBeggining,
                        BoardViewModelObject.Game.Board.NumberOfBlackPiecesAtBeggining,
                        BoardViewModelObject.Game.Variant,
                        BoardViewModelObject.Game.WhitePlayerEngine,
                        BoardViewModelObject.Game.BlackPlayerEngine,
                        BoardViewModelObject.MoveAnimationTime);
                }
                catch (NoAvailablePiecesException exception)
                {
                    BoardViewModelObject.DrawNextMove(exception.LastMove);
                    await this.ShowMessageAsync("Koniec gry", $"Gra zakończony. Gracz {(exception.Color == Logic.Enums.PieceColor.Black ? "CZARNY" : "BIAŁY")} nie ma już pionków.");
                    StartNewGame(
                        BoardViewModelObject.Game.Board.Size,
                        BoardViewModelObject.Game.Board.NumberOfWhitePiecesAtBeggining,
                        BoardViewModelObject.Game.Board.NumberOfBlackPiecesAtBeggining,
                        BoardViewModelObject.Game.Variant,
                        BoardViewModelObject.Game.WhitePlayerEngine,
                        BoardViewModelObject.Game.BlackPlayerEngine,
                        BoardViewModelObject.MoveAnimationTime);
                }
                catch (WrongMoveException exception)
                {
                    await this.ShowMessageAsync("Zły ruch", $"Gracz wykonał nielegalny ruch. Możliwa ilość pionków do bicia to {exception.MinimumBeatedPieces}.");

                }

            }


        }

        private PieceColor GetPlayerColorFromPath(Path path)
        {
            if (path.Fill == Brushes.White)
                return PieceColor.White;
            if (path.Fill == Brushes.Black)
                return PieceColor.Black;
            if (path.Fill == Brushes.Gold)
                return PieceColor.Black;
            throw new ArgumentException();
        }
        private void HistoryViewControl_Loaded(object sender, RoutedEventArgs e)
        {
            HistoryViewModelControl.DataContext = HistoryViewModelObject;
            HistoryListView = UiHelper.FindChild<ListView>(HistoryViewModelControl, "HistoryListView");
            HistoryListView.SelectionChanged += ListView_SelectionChanged;
        }

        private void ListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (HistoryListView.SelectedIndex == -1)
                return;
            if (HistoryListView.SelectedIndex == HistoryListView.Items.Count - 1)
                HistoryShowed = false;
            else
                HistoryShowed = true;
            BoardViewModelObject.DrawHistoryBoard(HistoryListView.SelectedIndex + 1);

        }

        private void MetroWindow_MouseMove(object sender, MouseEventArgs e)
        {
            IInputElement element = Mouse.DirectlyOver;
            if (element is Path)
            {

                var pathElem = (Path)element;
                if (pathElem.Data is RectangleGeometry || pathElem.Data is EllipseGeometry)
                {
                    var pos = BoardViewModelObject.BoardCanvasElements.SingleOrDefault(es => es.Geometry == pathElem.Data);
                    var position = Piece.ToPosition(pos.X, pos.Y, BoardViewModelObject.Game.Board.Size);
                    if (position == -1)
                    {
                        PositionLabel.Content = " ";
                    }
                    else
                    {
                        PositionLabel.Content = position.ToString();
                    }
                }
            }
            else
            {
                PositionLabel.Content = "?";
            }
        }

        private void SettingsButtonClick(object sender, RoutedEventArgs e)
        {
            if (settingsWindow != null)
            {
                settingsWindow.Activate();
                return;
            }

            settingsWindow = new SettingsWindow(
                this,
                BoardViewModelObject.Game.Board.Size,
                BoardViewModelObject.Game.Board.NumberOfWhitePiecesAtBeggining,
                BoardViewModelObject.Game.Board.NumberOfBlackPiecesAtBeggining,
                BoardViewModelObject.Game.Variant,
                BoardViewModelObject.Game.WhitePlayerEngine,
                BoardViewModelObject.Game.BlackPlayerEngine,
                BoardViewModelObject.MoveAnimationTime);
            settingsWindow.Owner = this;
            settingsWindow.Closed += (o, args) => settingsWindow = null;

            settingsWindow.Show();
        }
    }
}
