using Checkers.Logic.GameObjects;
using Checkers.UI.Model;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.UI.ViewModel
{
    public class HistoryViewModel
    {
        public HistoryViewModel()
        {
        }

        public ObservableCollection<History> History
        {
            get;
            set;
        } = new ObservableCollection<History>();
        
        public void AddHistoryItem(int size, Move move)
        {
            int fromNumber = GetCheckersPositionNumber(size, move.OldPiece.Row, move.OldPiece.Column);
            int toNumber = GetCheckersPositionNumber(size, move.NewPiece.Row, move.NewPiece.Column);


            if (move.BeatedPieces == null)
                History.Add(new History() { HistoryItem = $"  {History.Count + 1}.\t{fromNumber}-{toNumber}" });
            else
            {
                string numberString = "";
                foreach(var piece in move.BeatedPieces)
                {
                    numberString += $"{GetCheckersPositionNumber(size, piece.BeatPieceRow, piece.BeatPieceColumn)}x";
                }
                numberString += GetCheckersPositionNumber(size, move.NewPiece.Row, move.NewPiece.Column);
                History.Add(new History() { HistoryItem = $"  {History.Count + 1}.\t{numberString}" });
            }
        }

        private int GetCheckersPositionNumber(int size, int row, int column)
        {
            return size / 2 * (size - row - 1) + ((row % 2 == 0) ? 1 : 0) + (column + 1) / 2;
        }
    }
}
