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
        }

        BoardViewModel BboardViewModelObject = new BoardViewModel();

        public void BoardViewControl_Loaded(object sender, RoutedEventArgs e)
        {
            BboardViewModelObject.StartNewGame();
            BoardViewControl.DataContext = BboardViewModelObject;
        }

        private async void BoardViewControl_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            try
            {
                BboardViewModelObject.NextMove();
            }
            catch (NotAvailableMoveException exception)
            {
                await this.ShowMessageAsync("REMIS", $"Gra zakończona remisem gracz {(exception.Color == Logic.Enums.PieceColor.Black ? "CZARNY" : "BIAŁY")} nie może już wykonywać ruchów");
                BboardViewModelObject.StartNewGame();
            }
        }
    }
}
