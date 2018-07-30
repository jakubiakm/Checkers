using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;

namespace Checkers.UI.Model
{
    public class BoardModel { }
    public class CanvasElement : INotifyPropertyChanged
    {
        private Geometry geometry;

        private Brush stroke;

        private double row;

        private double column;

        private Brush fill;

        public double Row
        {
            get
            {
                return row;
            }
            set
            {
                row = value;
                RaisePropertyChanged("Row");
            }
        }
        public double Column
        {
            get
            {
                return column;
            }
            set
            {
                column = value;
                RaisePropertyChanged("Column");
            }
        }
        public Geometry Geometry
        {
            get
            {
                return geometry;
            }
            set
            {
                geometry = value;
                RaisePropertyChanged("Geometry");
            }
        }
        public Brush Stroke
        {
            get
            {
                return stroke;
            }
            set
            {
                stroke = value;
                RaisePropertyChanged("Stroke");
            }
        }
        public Brush Fill
        {
            get
            {
                return fill;
            }
            set
            {
                fill = value;
                RaisePropertyChanged("Fill");
            }
        }
        public double ActualHeight { get; set; }

        public int X { get; set; }

        public int Y { get; set; }

        public event PropertyChangedEventHandler PropertyChanged;

        private void RaisePropertyChanged(string property)
        {
            if (PropertyChanged != null)
            {
                PropertyChanged(this, new PropertyChangedEventArgs(property));
            }
        }
    }
    public class UiPiece : INotifyPropertyChanged
    {
        private string firstName;
        private string lastName;

        public string FirstName
        {
            get
            {
                return firstName;
            }

            set
            {
                if (firstName != value)
                {
                    firstName = value;
                    RaisePropertyChanged("FirstName");
                    RaisePropertyChanged("FullName");
                }
            }
        }

        public string LastName
        {
            get { return lastName; }

            set
            {
                if (lastName != value)
                {
                    lastName = value;
                    RaisePropertyChanged("LastName");
                    RaisePropertyChanged("FullName");
                }
            }
        }

        public string FullName
        {
            get
            {
                return firstName + " " + lastName;
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        private void RaisePropertyChanged(string property)
        {
            if (PropertyChanged != null)
            {
                PropertyChanged(this, new PropertyChangedEventArgs(property));
            }
        }
    }
}
