using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Markup;
using System.Windows.Media;
using Checkers.Logic.Engines;
using Checkers.Logic.Enums;
using Checkers.UI.ViewModel;
using MahApps.Metro;
using MahApps.Metro.Controls;

namespace Checkers.UI.Views
{
    [ValueConversion(typeof(Enum), typeof(IEnumerable<ValueDescription>))]
    public class EnumToCollectionConverter : MarkupExtension, IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return EnumHelper.GetAllValuesAndDescriptions(value.GetType());
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return null;
        }
        public override object ProvideValue(IServiceProvider serviceProvider)
        {
            return this;
        }
    }
    /// <summary>
    /// Interaction logic for AccentStyleWindow.xaml
    /// </summary>
    public partial class SettingsWindow : MetroWindow
    {
        public MainWindow Window { get; set; }

        SettingsViewModel viewModel;

        public static readonly DependencyProperty ColorsProperty
            = DependencyProperty.Register("Colors",
                                          typeof(List<KeyValuePair<string, Color>>),
                                          typeof(SettingsWindow),
                                          new PropertyMetadata(default(List<KeyValuePair<string, Color>>)));

        public List<KeyValuePair<string, Color>> Colors
        {
            get { return (List<KeyValuePair<string, Color>>)GetValue(ColorsProperty); }
            set { SetValue(ColorsProperty, value); }
        }

        public SettingsWindow(
            MainWindow window,
            int boardSize,
            int whiteCountSize,
            int blackCountSize,
            GameVariant gameVariant,
            IEngine whiteEngine,
            IEngine blackEngine,
            int moveAnimationTime)
        {
            this.Left = 0;
            this.Top = 0;

            Window = window;
            InitializeComponent();

            viewModel = new SettingsViewModel(
                boardSize,
                whiteCountSize,
                blackCountSize,
                whiteEngine,
                blackEngine,
                gameVariant,
                moveAnimationTime);

            this.DataContext = this;
            base.DataContext = viewModel;

            this.Colors = typeof(Colors)
                .GetProperties()
                .Where(prop => typeof(Color).IsAssignableFrom(prop.PropertyType))
                .Select(prop => new KeyValuePair<String, Color>(prop.Name, (Color)prop.GetValue(null)))
                .ToList();

            var theme = ThemeManager.DetectAppStyle(Application.Current);
            ThemeManager.ChangeAppStyle(this, theme.Item2, theme.Item1);
        }

        private void ChangeAppThemeButtonClick(object sender, RoutedEventArgs e)
        {
            var theme = ThemeManager.DetectAppStyle(Application.Current);
            ThemeManager.ChangeAppStyle(Application.Current, theme.Item2, ThemeManager.GetAppTheme("Base" + ((Button)sender).Content));
            ThemeManager.ChangeAppStyle(this, theme.Item2, ThemeManager.GetAppTheme("Base" + ((Button)sender).Content));
        }

        private void AccentSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var selectedAccent = AccentSelector.SelectedItem as Accent;
            if (selectedAccent != null)
            {
                var theme = ThemeManager.DetectAppStyle(Application.Current);
                ThemeManager.ChangeAppStyle(Application.Current, selectedAccent, theme.Item1);
                ThemeManager.ChangeAppStyle(this, selectedAccent, theme.Item1);
                Application.Current.MainWindow.Activate();
            }
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Dispatcher.Invoke(() =>
            {
                Application.Current.MainWindow.Activate();
                Window.StartNewGame(
                    viewModel.BoardSize,
                    viewModel.WhitePiecesCount,
                    viewModel.BlackPiecesCount,
                    viewModel.CurrentGameVariant,
                    ConvertEngineKindToEngine(viewModel.WhitePlayerEngineKind, PieceColor.White),
                    ConvertEngineKindToEngine(viewModel.BlackPlayerEngineKind, PieceColor.Black),
                    viewModel.MoveAnimationTime);
            });
        }

        private IEngine ConvertEngineKindToEngine(EngineKind engineKind, PieceColor color)
        {
            switch (engineKind)
            {
                case EngineKind.Human:
                    return new HumanEngine(color);
                case EngineKind.Random:
                    if (color == PieceColor.White)
                    {
                        return new RandomEngine(color, viewModel.WhitePlayerRandomEngineUseRandomSeed ? null : (int?)viewModel.WhitePlayerRandomEngineSeedValue);
                    }
                    else
                    {
                        return new RandomEngine(color, viewModel.BlackPlayerRandomEngineUseRandomSeed ? null : (int?)viewModel.BlackPlayerRandomEngineSeedValue);
                    }
                case EngineKind.Cuda:
                    if(color == PieceColor.White)
                    {
                        return new CudaEngine(color, viewModel.WhitePlayerCudaEngineMctsIteration, viewModel.WhitePlayerCudaEngineGridSize, viewModel.WhitePlayerCudaEngineBlockSize);
                    }
                    else
                    {
                        return new CudaEngine(color, viewModel.BlackPlayerCudaEngineMctsIteration, viewModel.BlackPlayerCudaEngineGridSize, viewModel.BlackPlayerCudaEngineBlockSize);
                    }
                case EngineKind.MinMax:
                    return new MinmaxEngine(color);
                default:
                    throw new ArgumentException("Nierozpoznany typ silnika");
            }
        }

        private void NumberValidationTextBox(object sender, System.Windows.Input.TextCompositionEventArgs e)
        {
            Regex regex = new Regex("[^0-9]");
            e.Handled = regex.IsMatch(e.Text);
        }

        private void BlackPlayerEngineCombobox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (BlackPlayerEngineCombobox.SelectedValue != null && (EngineKind)BlackPlayerEngineCombobox.SelectedValue == EngineKind.Cuda)
            {
                BoardSize.IsEnabled = false;
                viewModel.BlackPiecesCount = 20;
                viewModel.BoardSize = 10;
            }
            else
            {
                BoardSize.IsEnabled = true;
            }
        }

        private void WhitePlayerEngineCombobox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (WhitePlayerEngineCombobox.SelectedValue != null && (EngineKind)WhitePlayerEngineCombobox.SelectedValue == EngineKind.Cuda)
            {
                BoardSize.IsEnabled = false;
                viewModel.WhitePiecesCount = 20;
                viewModel.BoardSize = 10;
            }
            else
            {
                BoardSize.IsEnabled = true;
            }
        }
    }
}
