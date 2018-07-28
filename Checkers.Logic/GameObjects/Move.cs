using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.GameObjects
{
    public class Move
    {
        public Piece OldPiece { get; set; }

        public Piece NewPiece { get; set; }

        public List<Piece> BeatedPieces { get; set; }

        public Move(Piece oldPiece, Piece newPiece, List<Piece> beatedPieces)
        {
            OldPiece = oldPiece;
            NewPiece = newPiece;
            BeatedPieces = beatedPieces;
        }
    }
}
