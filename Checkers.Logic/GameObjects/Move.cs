using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.GameObjects
{
    [Serializable]
    public class Move
    {
        public Piece OldPiece { get; set; }

        public Piece NewPiece { get; set; }

        public List<BeatedPiece> BeatedPieces { get; set; }

        public Move(Piece oldPiece, Piece newPiece, List<BeatedPiece> beatedPieces)
        {
            OldPiece = oldPiece;
            NewPiece = newPiece;
            BeatedPieces = beatedPieces;
        }

        public string GetBeatedPiecesString()
        {
            string beatedPiecesString = "";
            if (BeatedPieces != null)
                foreach (var beatedPiece in BeatedPieces)
                {
                    beatedPiecesString += $"{beatedPiece.Position} ";
                }
            return beatedPiecesString;
        }
    }
}
