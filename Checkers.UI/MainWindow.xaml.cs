using Checkers.Logic;
using Checkers.Logic.Engines;
using Checkers.Logic.GameObjects;
using MahApps.Metro.Controls;
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
        public Game Game { get; set; }

        public MainWindow()
        {
            InitializeComponent();
            Game = new Game(new RandomEngine(), new RandomEngine());
        }
    }
}
