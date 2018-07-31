using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.UI.Model
{
    public class HistoryModel
    {
    }
    public class StudentModel { }

    public class History : INotifyPropertyChanged
    {
        private string historyItem;
        private string lastName;

        public string HistoryItem
        {
            get
            {
                return historyItem;
            }

            set
            {
                if (historyItem != value)
                {
                    historyItem = value;
                    RaisePropertyChanged("HistoryItem");
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
