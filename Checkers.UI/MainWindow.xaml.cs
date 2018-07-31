using Checkers.Logic.Exceptions;
using Checkers.UI.ViewModel;
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

namespace Checkers.UI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : MetroWindow
    {
        public MainWindow()
        {
            InitializeComponent();
            BoardViewModelObject = new BoardViewModel(this);
            HistoryViewModelObject = new HistoryViewModel();
        }

        Canvas BoardCanvas { get; set; }

        ListView HistoryListView { get; set; }

        bool HistoryShowed { get; set; } = false;

        BoardViewModel BoardViewModelObject;

        HistoryViewModel HistoryViewModelObject;

        public void BoardViewControl_Loaded(object sender, RoutedEventArgs e)
        {
            BoardViewModelObject.StartNewGame();
            BoardViewControl.DataContext = BoardViewModelObject;
            BoardCanvas = UiHelper.FindChild<Canvas>(BoardViewControl, "BoardCanvas");
            BoardCanvas.MouseDown += BoardCanvas_MouseDown;
        }

        private async void BoardCanvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            IInputElement clickedElement = Mouse.DirectlyOver;
            if (clickedElement is Path && !HistoryShowed)
            {
                HistoryListView.SelectedIndex = -1;
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
                    await this.ShowMessageAsync("Remis", $"Gra zakończona remisem gracz {(exception.Color == Logic.Enums.PieceColor.Black ? "CZARNY" : "BIAŁY")} nie może już wykonywać ruchów.");
                    HistoryViewModelObject.History.Clear();
                    BoardViewModelObject.StartNewGame();
                }
                catch (NoAvailablePiecesException exception)
                {
                    BoardViewModelObject.DrawNextMove(exception.LastMove);
                    await this.ShowMessageAsync("Koniec gry", $"Gra zakończony. Gracz {(exception.Color == Logic.Enums.PieceColor.Black ? "CZARNY" : "BIAŁY")} nie ma już pionków.");
                    HistoryViewModelObject.History.Clear();
                    BoardViewModelObject.StartNewGame();
                }

            }


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
    }
}
