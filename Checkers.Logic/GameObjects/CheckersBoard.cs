using Checkers.Logic.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.GameObjects
{
    [Serializable]
    public class CheckersBoard
    {
        public int Size { get; private set; } = 10;

        public List<Piece> PiecesOnBoard { get; private set; }

        public Move LastMove { get; set; }

        public int NumberOfWhitePiecesAtBeggining { get; set; }

        private int[] BoardArray { get; set; }

        public int NumberOfBlackPiecesAtBeggining { get; set; }

        private void SetBoardPosition(Piece piece)
        {
            if (piece.Color == PieceColor.White && !piece.IsKing)
                BoardArray[piece.Position] = 1;
            if (piece.Color == PieceColor.White && piece.IsKing)
                BoardArray[piece.Position] = 2;
            if (piece.Color == PieceColor.Black && !piece.IsKing)
                BoardArray[piece.Position] = -1;
            if (piece.Color == PieceColor.Black && piece.IsKing)
                BoardArray[piece.Position] = -2;
        }

        public CheckersBoard(int size, List<Piece> pieces)
        {
            BoardArray = new int[Size * Size + 1];

            foreach (var piece in pieces)
                SetBoardPosition(piece);

            Size = size;
            PiecesOnBoard = pieces;
            NumberOfBlackPiecesAtBeggining = pieces.Count(piece => piece.Color == PieceColor.Black);
            NumberOfWhitePiecesAtBeggining = pieces.Count(piece => piece.Color == PieceColor.White);
        }

        public CheckersBoard(int size, int numberOfWhitePieces, int numberOfBlackPieces)
        {
            BoardArray = new int[Size * Size + 1];
            NumberOfBlackPiecesAtBeggining = numberOfBlackPieces;
            NumberOfWhitePiecesAtBeggining = numberOfWhitePieces;
            PiecesOnBoard = new List<Piece>();
            Size = size;
            if (size * size / 2 < numberOfWhitePieces + numberOfBlackPieces)
            {
                throw new ArgumentException("Pionki nie mieszczą się na planszy");
            }
            if (size % 2 == 1)
            {
                throw new ArgumentException("Rozmiar planszy musi być liczbą parzystą");
            }
            while (numberOfWhitePieces-- > 0)
                PiecesOnBoard.Add(new Piece((size * size / 2) - numberOfWhitePieces, PieceColor.White, size, false));
            while (numberOfBlackPieces-- > 0)
                PiecesOnBoard.Add(new Piece(numberOfBlackPieces + 1, PieceColor.Black, size, false));
            foreach (var piece in PiecesOnBoard)
                SetBoardPosition(piece);
        }

        public List<Move> GetAllPossibleMoves(PieceColor color)
        {
            List<Move> possibleMoves = new List<Move>();
            int maxBeated = 0;
            foreach (var piece in PiecesOnBoard.Where(p => p.Color == color && p.IsKing == false).ToList())
            {
                possibleMoves.AddRange(GetPawnPossibleMoves(piece, ref maxBeated));
            }
            foreach (var piece in PiecesOnBoard.Where(p => p.Color == color && p.IsKing == true).ToList())
            {
                possibleMoves.AddRange(GetKingPossibleMoves(piece, ref maxBeated));
            }

            possibleMoves.ForEach(move => move.NewPiece.IsKing = move.OldPiece.IsKing || (move.NewPiece.Color == PieceColor.White ? move.NewPiece.Row == Size - 1 : move.NewPiece.Row == 0));
            var maximumBeatedPieces = possibleMoves.Count == 0 ? 0 : possibleMoves.Max(m => m.BeatedPieces?.Count ?? 0);
            return possibleMoves.Where(m => (m.BeatedPieces?.Count ?? 0) == maximumBeatedPieces)?.ToList() ?? new List<Move>();
        }

        public Move MakeMove(Move move)
        {
            int index = PiecesOnBoard.FindIndex(p => p.Row == move.OldPiece.Row && p.Column == move.OldPiece.Column);
            BoardArray[PiecesOnBoard[index].Position] = 0;
            PiecesOnBoard[index] = move.NewPiece;
            SetBoardPosition(move.NewPiece);
            foreach (var piece in move.BeatedPieces ?? new List<BeatedPiece>())
            {
                BoardArray[piece.Position] = 0;
                PiecesOnBoard.RemoveAll(p => p.Color == piece.Color && p.Column == piece.Column && p.Row == piece.Row && p.IsKing == piece.IsKing);
            }
            return move;
        }

        public CheckersBoard GetBoardAfterMove(Move move)
        {
            CheckersBoard ret = this.DeepClone();
            ret.MakeMove(move);
            return ret;
        }

        public char[] GetBoardArray()
        {
            char[] array = new char[Size * Size];
            foreach (var piece in PiecesOnBoard)
            {
                if (piece.Color == PieceColor.White && !piece.IsKing)
                {
                    array[piece.Position] = (char)1;
                }
                if (piece.Color == PieceColor.White && piece.IsKing)
                {
                    array[piece.Position] = (char)2;
                }
                if (piece.Color == PieceColor.Black && !piece.IsKing)
                {
                    array[piece.Position] = (char)3;
                }
                if (piece.Color == PieceColor.Black && piece.IsKing)
                {
                    array[piece.Position] = (char)4;
                }
            }
            return array;
        }

        private List<Move> GetKingPossibleMoves(Piece piece, ref int maxBeated)
        {
            List<Move> possibleMoves = new List<Move>();
            //normalne ruchy w czterech kierunkach aż do napotkania pionka lub końca planszy
            if (maxBeated == 0)
            {
                for (int ind = 1; ind < Size; ind++)
                {
                    if (CanMoveToPosition(piece.Row + ind, piece.Column + ind, piece))
                        possibleMoves.Add(new Move(piece, new Piece(piece.Row + ind, piece.Column + ind, piece.Color, piece.Size, piece.IsKing), null));
                    else
                        break;
                }
                for (int ind = 1; ind < Size; ind++)
                {
                    if (CanMoveToPosition(piece.Row + ind, piece.Column - ind, piece))
                        possibleMoves.Add(new Move(piece, new Piece(piece.Row + ind, piece.Column - ind, piece.Color, piece.Size, piece.IsKing), null));
                    else
                        break;
                }
                for (int ind = 1; ind < Size; ind++)
                {
                    if (CanMoveToPosition(piece.Row - ind, piece.Column + ind, piece))
                        possibleMoves.Add(new Move(piece, new Piece(piece.Row - ind, piece.Column + ind, piece.Color, piece.Size, piece.IsKing), null));
                    else
                        break;
                }
                for (int ind = 1; ind < Size; ind++)
                {
                    if (CanMoveToPosition(piece.Row - ind, piece.Column - ind, piece))
                        possibleMoves.Add(new Move(piece, new Piece(piece.Row - ind, piece.Column - ind, piece.Color, piece.Size, piece.IsKing), null));
                    else
                        break;
                }
            }
            //próba bicia w czterech różnych kierunkach damką
            for (int ind = 1; ind < Size; ind++)
                if (CanBeatPiece(piece, piece.Row - ind, piece.Column - ind, piece))
                {
                    maxBeated = 1;
                    var tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == piece.Row - ind && p.Column == piece.Column - ind);
                    BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, piece.Row, piece.Column);
                    GetAllKingBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row - ind - 1, piece.Column - ind - 1, ref possibleMoves);
                }
                else
                {
                    if (!CanMoveToPosition(piece.Row - ind, piece.Column - ind, piece))
                        break;
                }
            for (int ind = 1; ind < Size; ind++)
                if (CanBeatPiece(piece, piece.Row + ind, piece.Column - ind, piece))
                {
                    maxBeated = 1;
                    Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == piece.Row + ind && p.Column == piece.Column - ind);
                    BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, piece.Row, piece.Column);
                    GetAllKingBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row + ind + 1, piece.Column - ind - 1, ref possibleMoves);
                }
                else
                {
                    if (!CanMoveToPosition(piece.Row + ind, piece.Column - ind, piece))
                        break;
                }
            for (int ind = 1; ind < Size; ind++)
                if (CanBeatPiece(piece, piece.Row - ind, piece.Column + ind, piece))
                {
                    maxBeated = 1;
                    Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == piece.Row - ind && p.Column == piece.Column + ind);
                    BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, piece.Row, piece.Column);
                    GetAllKingBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row - ind - 1, piece.Column + ind + 1, ref possibleMoves);
                }
                else
                {
                    if (!CanMoveToPosition(piece.Row - ind, piece.Column + ind, piece))
                        break;
                }
            for (int ind = 1; ind < Size; ind++)
                if (CanBeatPiece(piece, piece.Row + ind, piece.Column + ind, piece))
                {
                    maxBeated = 1;
                    Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == piece.Row + ind && p.Column == piece.Column + ind);
                    BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, piece.Row, piece.Column);
                    GetAllKingBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row + ind + 1, piece.Column + ind + 1, ref possibleMoves);
                }
                else
                {
                    if (!CanMoveToPosition(piece.Row + ind, piece.Column + ind, piece))
                        break;
                }
            return possibleMoves;
        }

        private List<Move> GetPawnPossibleMoves(Piece piece, ref int maxBeated)
        {
            List<Move> possibleMoves = new List<Move>();
            //normalne ruchy do przodu
            if (maxBeated == 0)
            {
                switch (piece.Color)
                {
                    case PieceColor.White:
                        if (CanMoveToPosition(piece.Row + 1, piece.Column + 1, piece))
                            possibleMoves.Add(new Move(piece, new Piece(piece.Row + 1, piece.Column + 1, PieceColor.White, piece.Size, piece.Row + 1 == Size - 1), null));
                        if (CanMoveToPosition(piece.Row + 1, piece.Column - 1, piece))
                            possibleMoves.Add(new Move(piece, new Piece(piece.Row + 1, piece.Column - 1, PieceColor.White, piece.Size, piece.Row + 1 == Size - 1), null));
                        break;
                    case PieceColor.Black:
                        if (CanMoveToPosition(piece.Row - 1, piece.Column + 1, piece))
                            possibleMoves.Add(new Move(piece, new Piece(piece.Row - 1, piece.Column + 1, PieceColor.Black, piece.Size, piece.Row - 1 == 0), null));
                        if (CanMoveToPosition(piece.Row - 1, piece.Column - 1, piece))
                            possibleMoves.Add(new Move(piece, new Piece(piece.Row - 1, piece.Column - 1, PieceColor.Black, piece.Size, piece.Row - 1 == 0), null));
                        break;
                }
            }
            //próba bicia w czterech różnych kierunkach
            if (CanBeatPiece(piece, piece.Row - 1, piece.Column - 1, piece))
            {
                maxBeated = 1;
                Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == piece.Row - 1 && p.Column == piece.Column - 1);
                BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, piece.Row, piece.Column);
                GetAllBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row - 2, piece.Column - 2, ref possibleMoves);
            }
            if (CanBeatPiece(piece, piece.Row + 1, piece.Column - 1, piece))
            {
                maxBeated = 1;
                Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == piece.Row + 1 && p.Column == piece.Column - 1);
                BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, piece.Row, piece.Column);
                GetAllBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row + 2, piece.Column - 2, ref possibleMoves);
            }
            if (CanBeatPiece(piece, piece.Row - 1, piece.Column + 1, piece))
            {
                maxBeated = 1;
                Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == piece.Row - 1 && p.Column == piece.Column + 1);
                BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, piece.Row, piece.Column);
                GetAllBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row - 2, piece.Column + 2, ref possibleMoves);
            }
            if (CanBeatPiece(piece, piece.Row + 1, piece.Column + 1, piece))
            {
                maxBeated = 1;
                Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == piece.Row + 1 && p.Column == piece.Column + 1);
                BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, piece.Row, piece.Column);
                GetAllBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row + 2, piece.Column + 2, ref possibleMoves);
            }
            return possibleMoves;
        }

        /// <summary>
        /// Funkcja sprawdzająca, czy można się ruszyć na daną pozycję na planszy
        /// </summary>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <param name="sourceMovePiece"></param>
        /// <returns></returns>
        private bool CanMoveToPosition(int row, int column, Piece sourceMovePiece)
        {
            int pos = Piece.ToPosition(row, column, Size);
            return
                row >= 0 && row < Size && column >= 0 && column < Size &&
                BoardArray[pos] == 0;
        }

        private bool CanBeatPiece(Piece piece, int row, int column, Piece sourceMovePiece)
        {
            int rowAfterBeat = row + (row - piece.Row > 0 ? 1 : -1);
            int columnAfterBeat = column + (column - piece.Column > 0 ? 1 : -1);
            int positionAfterBeat = Piece.ToPosition(rowAfterBeat, columnAfterBeat, Size);
            int position = Piece.ToPosition(row, column, Size);
            int piecePosition = Piece.ToPosition(sourceMovePiece.Row, sourceMovePiece.Column, Size);
            //sprawdzenie czy bite pole i pole po biciu mieszczą się w planszy
            if (!(row >= 0 && row < Size && column >= 0 && column < Size &&
                rowAfterBeat >= 0 && rowAfterBeat < Size && columnAfterBeat >= 0 && columnAfterBeat < Size))
                return false;
            //sprawdzenie czy jest przeciwny pionek na pozycji i czy po biciu można postawić pionka na następnym polu
            if (piece.Color == PieceColor.White)
            {
                if (BoardArray[position] < 0 && (BoardArray[positionAfterBeat] == 0 || positionAfterBeat == piecePosition))
                    return true;
                return false;
            }
            if (piece.Color == PieceColor.Black)
            {
                if (BoardArray[position] > 0 && (BoardArray[positionAfterBeat] == 0 || positionAfterBeat == piecePosition))
                    return true;
                return false;
            }
            if (PiecesOnBoard
                .Where(p => p.Position != sourceMovePiece.Position) //bijący pionek nie powinien być branyc pod uwagę
                .Count(p => p.Row == row && p.Column == column && p.Color != piece.Color) > 0 &&
                !PiecesOnBoard
                .Where(p => p.Position != sourceMovePiece.Position) //bijący pionek nie powinien być branyc pod uwagę
                .Any(p => p.Row == rowAfterBeat && p.Column == columnAfterBeat))
                return true;
            return false;
        }

        private void GetAllBeatMoves(Piece piece, List<BeatedPiece> beatedPieces, int sourceRow, int sourceColumn, int targetRow, int targetColumn, ref List<Move> allMoves)
        {
            Piece newPiece = new Piece(targetRow, targetColumn, piece.Color, piece.Size, piece.IsKing);
            allMoves.Add(new Move(piece, newPiece, beatedPieces));
            if (CanBeatPiece(newPiece, targetRow - 1, targetColumn - 1, piece))
            {
                Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == targetRow - 1 && p.Column == targetColumn - 1);
                BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, targetRow, targetColumn);
                if (beatedPieces.Where(p => p.Row == beatedPiece.Row && p.Column == beatedPiece.Column && p.Color == beatedPiece.Color).Count() == 0)
                {
                    List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                    newBeatedPieces.Add(beatedPiece);
                    GetAllBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow - 2, targetColumn - 2, ref allMoves);
                }
            }
            if (CanBeatPiece(newPiece, targetRow + 1, targetColumn - 1, piece))
            {
                Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == targetRow + 1 && p.Column == targetColumn - 1);
                BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, targetRow, targetColumn);
                if (beatedPieces.Where(p => p.Row == beatedPiece.Row && p.Column == beatedPiece.Column && p.Color == beatedPiece.Color).Count() == 0)
                {
                    List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                    newBeatedPieces.Add(beatedPiece);
                    GetAllBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow + 2, targetColumn - 2, ref allMoves);
                }
            }
            if (CanBeatPiece(newPiece, targetRow - 1, targetColumn + 1, piece))
            {
                Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == targetRow - 1 && p.Column == targetColumn + 1);
                BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, targetRow, targetColumn);
                if (beatedPieces.Where(p => p.Row == beatedPiece.Row && p.Column == beatedPiece.Column && p.Color == beatedPiece.Color).Count() == 0)
                {
                    List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                    newBeatedPieces.Add(beatedPiece);
                    GetAllBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow - 2, targetColumn + 2, ref allMoves);
                }
            }
            if (CanBeatPiece(newPiece, targetRow + 1, targetColumn + 1, piece))
            {
                Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == targetRow + 1 && p.Column == targetColumn + 1);
                BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, targetRow, targetColumn);
                if (beatedPieces.Where(p => p.Row == beatedPiece.Row && p.Column == beatedPiece.Column && p.Color == beatedPiece.Color).Count() == 0)
                {
                    List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                    newBeatedPieces.Add(beatedPiece);
                    GetAllBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow + 2, targetColumn + 2, ref allMoves);
                }
            }
        }

        private void GetAllKingBeatMoves(Piece piece, List<BeatedPiece> beatedPieces, int sourceRow, int sourceColumn, int targetRow, int targetColumn, ref List<Move> allMoves)
        {
            Piece newPiece = new Piece(targetRow, targetColumn, piece.Color, piece.Size, piece.IsKing);
            allMoves.Add(new Move(piece, newPiece, beatedPieces));
            for (int ind = 1; ind < Size; ind++)
            {
                if (targetRow - sourceRow > 0 && targetColumn - sourceColumn > 0)
                    if (CanMoveToPosition(targetRow + ind, targetColumn + ind, piece))
                        GetAllKingBeatMoves(piece, new List<BeatedPiece>(beatedPieces), targetRow, targetColumn, targetRow + ind, targetColumn + ind, ref allMoves);
                    else
                        break;
                if (targetRow - sourceRow > 0 && targetColumn - sourceColumn < 0)
                    if (CanMoveToPosition(targetRow + ind, targetColumn - ind, piece))
                        GetAllKingBeatMoves(piece, new List<BeatedPiece>(beatedPieces), targetRow, targetColumn, targetRow + ind, targetColumn - ind, ref allMoves);
                    else
                        break;
                if (targetRow - sourceRow < 0 && targetColumn - sourceColumn > 0)
                    if (CanMoveToPosition(targetRow - ind, targetColumn + ind, piece))
                        GetAllKingBeatMoves(piece, new List<BeatedPiece>(beatedPieces), targetRow, targetColumn, targetRow - ind, targetColumn + ind, ref allMoves);
                    else
                        break;
                if (targetRow - sourceRow < 0 && targetColumn - sourceColumn < 0)
                    if (CanMoveToPosition(targetRow - ind, targetColumn - ind, piece))
                        GetAllKingBeatMoves(piece, new List<BeatedPiece>(beatedPieces), targetRow, targetColumn, targetRow - ind, targetColumn - ind, ref allMoves);
                    else
                        break;
            }
            if (!(targetRow - sourceRow > 0 && targetColumn - sourceColumn > 0))
                for (int ind = 1; ind < Size; ind++)
                    if (CanBeatPiece(newPiece, targetRow - ind, targetColumn - ind, piece))
                    {
                        Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == targetRow - ind && p.Column == targetColumn - ind);
                        BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, targetRow, targetColumn);
                        if (beatedPieces.Where(p => p.Position == beatedPiece.Position && p.Color == beatedPiece.Color).Count() == 0)
                        {
                            List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                            newBeatedPieces.Add(beatedPiece);
                            GetAllKingBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow - ind - 1, targetColumn - ind - 1, ref allMoves);
                            break;
                        }
                        else
                            break;
                    }
                    else
                    {
                        if (!CanMoveToPosition(targetRow - ind, targetColumn - ind, piece))
                            break;
                    }
            if (!(targetRow - sourceRow < 0 && targetColumn - sourceColumn > 0))
                for (int ind = 1; ind < Size; ind++)
                    if (CanBeatPiece(newPiece, targetRow + ind, targetColumn - ind, piece))
                    {
                        Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == targetRow + ind && p.Column == targetColumn - ind);
                        BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, targetRow, targetColumn);
                        if (beatedPieces.Where(p => p.Position == beatedPiece.Position && p.Color == beatedPiece.Color).Count() == 0)
                        {
                            List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                            newBeatedPieces.Add(beatedPiece);
                            GetAllKingBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow + ind + 1, targetColumn - ind - 1, ref allMoves);
                            break;
                        }
                        else
                            break;
                    }
                    else
                    {
                        if (!CanMoveToPosition(targetRow + ind, targetColumn - ind, piece))
                            break;
                    }
            if (!(targetRow - sourceRow > 0 && targetColumn - sourceColumn < 0))
                for (int ind = 1; ind < Size; ind++)
                    if (CanBeatPiece(newPiece, targetRow - ind, targetColumn + ind, piece))
                    {
                        Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == targetRow - ind && p.Column == targetColumn + ind);
                        BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, targetRow, targetColumn);
                        if (beatedPieces.Where(p => p.Position == beatedPiece.Position && p.Color == beatedPiece.Color).Count() == 0)
                        {
                            List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                            newBeatedPieces.Add(beatedPiece);
                            GetAllKingBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow - ind - 1, targetColumn + ind + 1, ref allMoves);
                            break;
                        }
                        else
                            break;
                    }
                    else
                    {
                        if (!CanMoveToPosition(targetRow - ind, targetColumn + ind, piece))
                            break;
                    }
            if (!(targetRow - sourceRow < 0 && targetColumn - sourceColumn < 0))
                for (int ind = 1; ind < Size; ind++)
                    if (CanBeatPiece(newPiece, targetRow + ind, targetColumn + ind, piece))
                    {
                        Piece tempPiece = PiecesOnBoard.SingleOrDefault(p => p.Row == targetRow + ind && p.Column == targetColumn + ind);
                        BeatedPiece beatedPiece = new BeatedPiece(tempPiece.Row, tempPiece.Column, tempPiece.Color, tempPiece.IsKing, targetRow, targetColumn);
                        if (beatedPieces.Where(p => p.Position == beatedPiece.Position && p.Color == beatedPiece.Color).Count() == 0)
                        {
                            List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                            newBeatedPieces.Add(beatedPiece);
                            GetAllKingBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow + ind + 1, targetColumn + ind + 1, ref allMoves);
                            break;
                        }
                        else
                            break;
                    }
                    else
                    {
                        if (!CanMoveToPosition(targetRow + ind, targetColumn + ind, piece))
                            break;
                    }
        }

        public override string ToString()
        {
            string boardString = "";
            foreach (var piece in PiecesOnBoard)
            {
                boardString += $"{piece.Position}:{(piece.Color == PieceColor.White ? 1 : -1) * (piece.IsKing ? 2 : 1)} ";
            }
            return boardString;
        }
    }
}
