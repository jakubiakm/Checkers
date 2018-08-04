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
            int fromNumber = move.OldPiece.Position;
            int toNumber = move.NewPiece.Position;


            if (move.BeatedPieces == null)
                History.Add(new History() { HistoryItem = $"  {History.Count + 1}.\t{fromNumber}-{toNumber}" });
            else
            {
                string numberString = "";
                foreach(var piece in move.BeatedPieces)
                {
                    numberString += $"{Piece.ToPosition(piece.BeatPieceRow, piece.BeatPieceColumn, size)}x";
                }
                numberString += move.NewPiece.Position;
                History.Add(new History() { HistoryItem = $"  {History.Count + 1}.\t{numberString}" });
            }
        }
    }
}
