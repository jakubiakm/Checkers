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

        public Move LastMove { get; set; }

        public int NumberOfWhitePiecesAtBeggining { get; set; }

        public int[] BoardArray { get; set; }

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
            NumberOfBlackPiecesAtBeggining = pieces.Count(piece => piece.Color == PieceColor.Black);
            NumberOfWhitePiecesAtBeggining = pieces.Count(piece => piece.Color == PieceColor.White);
        }

        public CheckersBoard(int size, int numberOfWhitePieces, int numberOfBlackPieces)
        {
            BoardArray = new int[Size * Size + 1];
            NumberOfBlackPiecesAtBeggining = numberOfBlackPieces;
            NumberOfWhitePiecesAtBeggining = numberOfWhitePieces;
            var PiecesOnBoard = new List<Piece>();
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
            for (int i = 0; i != Size * Size + 1; i++)
            {
                switch (color)
                {
                    case PieceColor.White:
                        if (BoardArray[i] == 1)
                            possibleMoves.AddRange(GetPawnPossibleMoves(i, PieceColor.White, ref maxBeated));
                        if (BoardArray[i] == 2)
                            possibleMoves.AddRange(GetKingPossibleMoves(i, PieceColor.White, ref maxBeated));
                        break;
                    case PieceColor.Black:
                        if (BoardArray[i] == -1)
                            possibleMoves.AddRange(GetPawnPossibleMoves(i, PieceColor.Black, ref maxBeated));
                        if (BoardArray[i] == -2)
                            possibleMoves.AddRange(GetKingPossibleMoves(i, PieceColor.Black, ref maxBeated));
                        break;
                }
            }

            var maximumBeatedPieces = possibleMoves.Count == 0 ? 0 : possibleMoves.Max(m => m.BeatedPieces?.Count ?? 0);
            possibleMoves = possibleMoves.Where(m => (m.BeatedPieces?.Count ?? 0) == maximumBeatedPieces)?.ToList() ?? new List<Move>();
            possibleMoves.ForEach(move => move.NewPiece.IsKing = move.OldPiece.IsKing || (move.NewPiece.Color == PieceColor.White ? move.NewPiece.Row == Size - 1 : move.NewPiece.Row == 0));
            for (int i = 0; i != possibleMoves.Count; i++)
            {
                for (int j = 0; j != possibleMoves.Count; j++)
                {
                    if (i != j)
                    {
                        if (possibleMoves[i].NewPiece.Position == possibleMoves[j].NewPiece.Position &&
                            possibleMoves[i].OldPiece.Position == possibleMoves[j].OldPiece.Position)
                            possibleMoves.RemoveAt(j--);
                    }
                }
            }
            return possibleMoves;
        }

        public Move MakeMove(Move move)
        {
            int index = move.OldPiece.Position;
            BoardArray[index] = 0;
            SetBoardPosition(move.NewPiece);
            foreach (var piece in move.BeatedPieces ?? new List<BeatedPiece>())
            {
                BoardArray[piece.Position] = 0;
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

            for (int i = 0; i != Size * Size + 1; i++)
            {
                if (BoardArray[i] == 1)
                {
                    array[i] = (char)1;
                }
                if (BoardArray[i] == 2)
                {
                    array[i] = (char)2;
                }
                if (BoardArray[i] == -1)
                {
                    array[i] = (char)3;
                }
                if (BoardArray[i] == -2)
                {
                    array[i] = (char)4;
                }
            }
            return array;
        }

        private List<Move> GetKingPossibleMoves(int piecePosition, PieceColor color, ref int maxBeated)
        {
            var piece = new Piece(piecePosition, color, Size, true);
            List<Move> possibleMoves = new List<Move>();
            //normalne ruchy w czterech kierunkach aż do napotkania pionka lub końca planszy
            if (maxBeated == 0)
            {
                int pieceRow = Piece.ToRow(piecePosition, Size);
                int pieceColumn = Piece.ToColumn(piecePosition, Size);

                for (int ind = 1; ind < Size; ind++)
                {
                    if (CanMoveToPosition(pieceRow + ind, pieceColumn + ind, piecePosition))
                        possibleMoves.Add(new Move(piece, new Piece(piece.Row + ind, piece.Column + ind, piece.Color, piece.Size, piece.IsKing), null));
                    else
                        break;
                }
                for (int ind = 1; ind < Size; ind++)
                {
                    if (CanMoveToPosition(piece.Row + ind, piece.Column - ind, piecePosition))
                        possibleMoves.Add(new Move(piece, new Piece(piece.Row + ind, piece.Column - ind, piece.Color, piece.Size, piece.IsKing), null));
                    else
                        break;
                }
                for (int ind = 1; ind < Size; ind++)
                {
                    if (CanMoveToPosition(piece.Row - ind, piece.Column + ind, piecePosition))
                        possibleMoves.Add(new Move(piece, new Piece(piece.Row - ind, piece.Column + ind, piece.Color, piece.Size, piece.IsKing), null));
                    else
                        break;
                }
                for (int ind = 1; ind < Size; ind++)
                {
                    if (CanMoveToPosition(piece.Row - ind, piece.Column - ind, piecePosition))
                        possibleMoves.Add(new Move(piece, new Piece(piece.Row - ind, piece.Column - ind, piece.Color, piece.Size, piece.IsKing), null));
                    else
                        break;
                }
            }
            //próba bicia w czterech różnych kierunkach damką
            for (int ind = 1; ind < Size; ind++)
                if (CanBeatPiece(piece, piece.Row - ind, piece.Column - ind, piece))
                {
                    BeatedPiece beatedPiece = new BeatedPiece(piece.Row - ind, piece.Column - ind, color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, piece.Row, piece.Column);
                    GetAllKingBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row - ind - 1, piece.Column - ind - 1, ref possibleMoves, ref maxBeated);
                }
                else
                {
                    if (!CanMoveToPosition(piece.Row - ind, piece.Column - ind, piecePosition))
                        break;
                }
            for (int ind = 1; ind < Size; ind++)
                if (CanBeatPiece(piece, piece.Row + ind, piece.Column - ind, piece))
                {
                    BeatedPiece beatedPiece = new BeatedPiece(piece.Row + ind, piece.Column - ind, color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, piece.Row, piece.Column);
                    GetAllKingBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row + ind + 1, piece.Column - ind - 1, ref possibleMoves, ref maxBeated);
                }
                else
                {
                    if (!CanMoveToPosition(piece.Row + ind, piece.Column - ind, piecePosition))
                        break;
                }
            for (int ind = 1; ind < Size; ind++)
                if (CanBeatPiece(piece, piece.Row - ind, piece.Column + ind, piece))
                {
                    BeatedPiece beatedPiece = new BeatedPiece(piece.Row - ind, piece.Column + ind, color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, piece.Row, piece.Column);
                    GetAllKingBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row - ind - 1, piece.Column + ind + 1, ref possibleMoves, ref maxBeated);
                }
                else
                {
                    if (!CanMoveToPosition(piece.Row - ind, piece.Column + ind, piecePosition))
                        break;
                }
            for (int ind = 1; ind < Size; ind++)
                if (CanBeatPiece(piece, piece.Row + ind, piece.Column + ind, piece))
                {
                    BeatedPiece beatedPiece = new BeatedPiece(piece.Row + ind, piece.Column + ind, color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, piece.Row, piece.Column);
                    GetAllKingBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row + ind + 1, piece.Column + ind + 1, ref possibleMoves, ref maxBeated);
                }
                else
                {
                    if (!CanMoveToPosition(piece.Row + ind, piece.Column + ind, piecePosition))
                        break;
                }
            return possibleMoves;
        }

        private List<Move> GetPawnPossibleMoves(int piecePosition, PieceColor color, ref int maxBeated)
        {
            List<Move> possibleMoves = new List<Move>();
            var piece = new Piece(piecePosition, color, Size, false);
            //normalne ruchy do przodu
            if (maxBeated == 0)
            {
                switch (piece.Color)
                {
                    case PieceColor.White:
                        if (CanMoveToPosition(piece.Row + 1, piece.Column + 1, piecePosition))
                            possibleMoves.Add(new Move(piece, new Piece(piece.Row + 1, piece.Column + 1, PieceColor.White, piece.Size, piece.Row + 1 == Size - 1), null));
                        if (CanMoveToPosition(piece.Row + 1, piece.Column - 1, piecePosition))
                            possibleMoves.Add(new Move(piece, new Piece(piece.Row + 1, piece.Column - 1, PieceColor.White, piece.Size, piece.Row + 1 == Size - 1), null));
                        break;
                    case PieceColor.Black:
                        if (CanMoveToPosition(piece.Row - 1, piece.Column + 1, piecePosition))
                            possibleMoves.Add(new Move(piece, new Piece(piece.Row - 1, piece.Column + 1, PieceColor.Black, piece.Size, piece.Row - 1 == 0), null));
                        if (CanMoveToPosition(piece.Row - 1, piece.Column - 1, piecePosition))
                            possibleMoves.Add(new Move(piece, new Piece(piece.Row - 1, piece.Column - 1, PieceColor.Black, piece.Size, piece.Row - 1 == 0), null));
                        break;
                }
            }
            //próba bicia w czterech różnych kierunkach
            if (CanBeatPiece(piece, piece.Row - 1, piece.Column - 1, piece))
            {
                BeatedPiece beatedPiece = new BeatedPiece(piece.Row - 1, piece.Column - 1, piece.Color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, piece.Row, piece.Column);
                GetAllBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row - 2, piece.Column - 2, ref possibleMoves, ref maxBeated);
            }
            if (CanBeatPiece(piece, piece.Row + 1, piece.Column - 1, piece))
            {
                BeatedPiece beatedPiece = new BeatedPiece(piece.Row + 1, piece.Column - 1, piece.Color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, piece.Row, piece.Column);
                GetAllBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row + 2, piece.Column - 2, ref possibleMoves, ref maxBeated);
            }
            if (CanBeatPiece(piece, piece.Row - 1, piece.Column + 1, piece))
            {
                BeatedPiece beatedPiece = new BeatedPiece(piece.Row - 1, piece.Column + 1, piece.Color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, piece.Row, piece.Column);
                GetAllBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row - 2, piece.Column + 2, ref possibleMoves, ref maxBeated);
            }
            if (CanBeatPiece(piece, piece.Row + 1, piece.Column + 1, piece))
            {
                BeatedPiece beatedPiece = new BeatedPiece(piece.Row + 1, piece.Column + 1, piece.Color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, piece.Row, piece.Column);
                GetAllBeatMoves(piece, new List<BeatedPiece>() { beatedPiece }, piece.Row, piece.Column, piece.Row + 2, piece.Column + 2, ref possibleMoves, ref maxBeated);
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
        private bool CanMoveToPosition(int row, int column, int pieceSourcePosition)
        {
            int pos = Piece.ToPosition(row, column, Size);
            return
                row >= 0 && row < Size && column >= 0 && column < Size &&
                (BoardArray[pos] == 0 || pos == pieceSourcePosition);
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
            return false;
        }

        private void GetAllBeatMoves(Piece piece, List<BeatedPiece> beatedPieces, int sourceRow, int sourceColumn, int targetRow, int targetColumn, ref List<Move> allMoves, ref int maxBeated)
        {
            if (beatedPieces.Count > maxBeated)
                maxBeated = beatedPieces.Count;
            Piece newPiece = new Piece(targetRow, targetColumn, piece.Color, piece.Size, piece.IsKing);
            if (beatedPieces.Count == maxBeated)
                allMoves.Add(new Move(piece, newPiece, beatedPieces));
            if (CanBeatPiece(newPiece, targetRow - 1, targetColumn - 1, piece))
            {
                BeatedPiece beatedPiece = new BeatedPiece(targetRow - 1, targetColumn - 1, piece.Color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, targetRow, targetColumn);
                if (beatedPieces.Where(p => p.Row == beatedPiece.Row && p.Column == beatedPiece.Column && p.Color == beatedPiece.Color).Count() == 0)
                {
                    List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                    newBeatedPieces.Add(beatedPiece);
                    GetAllBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow - 2, targetColumn - 2, ref allMoves, ref maxBeated);
                }
            }
            if (CanBeatPiece(newPiece, targetRow + 1, targetColumn - 1, piece))
            {
                BeatedPiece beatedPiece = new BeatedPiece(targetRow + 1, targetColumn - 1, piece.Color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, targetRow, targetColumn);
                if (beatedPieces.Where(p => p.Row == beatedPiece.Row && p.Column == beatedPiece.Column && p.Color == beatedPiece.Color).Count() == 0)
                {
                    List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                    newBeatedPieces.Add(beatedPiece);
                    GetAllBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow + 2, targetColumn - 2, ref allMoves, ref maxBeated);
                }
            }
            if (CanBeatPiece(newPiece, targetRow - 1, targetColumn + 1, piece))
            {
                BeatedPiece beatedPiece = new BeatedPiece(targetRow - 1, targetColumn + 1, piece.Color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, targetRow, targetColumn);
                if (beatedPieces.Where(p => p.Row == beatedPiece.Row && p.Column == beatedPiece.Column && p.Color == beatedPiece.Color).Count() == 0)
                {
                    List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                    newBeatedPieces.Add(beatedPiece);
                    GetAllBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow - 2, targetColumn + 2, ref allMoves, ref maxBeated);
                }
            }
            if (CanBeatPiece(newPiece, targetRow + 1, targetColumn + 1, piece))
            {
                BeatedPiece beatedPiece = new BeatedPiece(targetRow + 1, targetColumn + 1, piece.Color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, targetRow, targetColumn);
                if (beatedPieces.Where(p => p.Row == beatedPiece.Row && p.Column == beatedPiece.Column && p.Color == beatedPiece.Color).Count() == 0)
                {
                    List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                    newBeatedPieces.Add(beatedPiece);
                    GetAllBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow + 2, targetColumn + 2, ref allMoves, ref maxBeated);
                }
            }
        }

        private void GetAllKingBeatMoves(Piece piece, List<BeatedPiece> beatedPieces, int sourceRow, int sourceColumn, int targetRow, int targetColumn, ref List<Move> allMoves, ref int maxBeated)
        {
            if (beatedPieces.Count > maxBeated)
                maxBeated = beatedPieces.Count;
            Piece newPiece = new Piece(targetRow, targetColumn, piece.Color, piece.Size, piece.IsKing);
            if (beatedPieces.Count == maxBeated)
                allMoves.Add(new Move(piece, newPiece, beatedPieces));
            for (int ind = 1; ind < Size; ind++)
            {
                if (targetRow - sourceRow > 0 && targetColumn - sourceColumn > 0)
                    if (CanMoveToPosition(targetRow + ind, targetColumn + ind, piece.Position))
                        GetAllKingBeatMoves(piece, new List<BeatedPiece>(beatedPieces), targetRow, targetColumn, targetRow + ind, targetColumn + ind, ref allMoves, ref maxBeated);
                    else
                        break;
                if (targetRow - sourceRow > 0 && targetColumn - sourceColumn < 0)
                    if (CanMoveToPosition(targetRow + ind, targetColumn - ind, piece.Position))
                        GetAllKingBeatMoves(piece, new List<BeatedPiece>(beatedPieces), targetRow, targetColumn, targetRow + ind, targetColumn - ind, ref allMoves, ref maxBeated);
                    else
                        break;
                if (targetRow - sourceRow < 0 && targetColumn - sourceColumn > 0)
                    if (CanMoveToPosition(targetRow - ind, targetColumn + ind, piece.Position))
                        GetAllKingBeatMoves(piece, new List<BeatedPiece>(beatedPieces), targetRow, targetColumn, targetRow - ind, targetColumn + ind, ref allMoves, ref maxBeated);
                    else
                        break;
                if (targetRow - sourceRow < 0 && targetColumn - sourceColumn < 0)
                    if (CanMoveToPosition(targetRow - ind, targetColumn - ind, piece.Position))
                        GetAllKingBeatMoves(piece, new List<BeatedPiece>(beatedPieces), targetRow, targetColumn, targetRow - ind, targetColumn - ind, ref allMoves, ref maxBeated);
                    else
                        break;
            }
            if (!(targetRow - sourceRow > 0 && targetColumn - sourceColumn > 0))
                for (int ind = 1; ind < Size; ind++)
                    if (CanBeatPiece(newPiece, targetRow - ind, targetColumn - ind, piece))
                    {
                        BeatedPiece beatedPiece = new BeatedPiece(targetRow - ind, targetColumn - ind, piece.Color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, targetRow, targetColumn);
                        if (beatedPieces.Where(p => p.Position == beatedPiece.Position && p.Color == beatedPiece.Color).Count() == 0)
                        {
                            List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                            newBeatedPieces.Add(beatedPiece);
                            GetAllKingBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow - ind - 1, targetColumn - ind - 1, ref allMoves, ref maxBeated);
                            break;
                        }
                        else
                            break;
                    }
                    else
                    {
                        if (!CanMoveToPosition(targetRow - ind, targetColumn - ind, piece.Position))
                            break;
                    }
            if (!(targetRow - sourceRow < 0 && targetColumn - sourceColumn > 0))
                for (int ind = 1; ind < Size; ind++)
                    if (CanBeatPiece(newPiece, targetRow + ind, targetColumn - ind, piece))
                    {
                        BeatedPiece beatedPiece = new BeatedPiece(targetRow + ind, targetColumn - ind, piece.Color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, targetRow, targetColumn);
                        if (beatedPieces.Where(p => p.Position == beatedPiece.Position && p.Color == beatedPiece.Color).Count() == 0)
                        {
                            List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                            newBeatedPieces.Add(beatedPiece);
                            GetAllKingBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow + ind + 1, targetColumn - ind - 1, ref allMoves, ref maxBeated);
                            break;
                        }
                        else
                            break;
                    }
                    else
                    {
                        if (!CanMoveToPosition(targetRow + ind, targetColumn - ind, piece.Position))
                            break;
                    }
            if (!(targetRow - sourceRow > 0 && targetColumn - sourceColumn < 0))
                for (int ind = 1; ind < Size; ind++)
                    if (CanBeatPiece(newPiece, targetRow - ind, targetColumn + ind, piece))
                    {
                        BeatedPiece beatedPiece = new BeatedPiece(targetRow - ind, targetColumn + ind, piece.Color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, targetRow, targetColumn);
                        if (beatedPieces.Where(p => p.Position == beatedPiece.Position && p.Color == beatedPiece.Color).Count() == 0)
                        {
                            List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                            newBeatedPieces.Add(beatedPiece);
                            GetAllKingBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow - ind - 1, targetColumn + ind + 1, ref allMoves, ref maxBeated);
                            break;
                        }
                        else
                            break;
                    }
                    else
                    {
                        if (!CanMoveToPosition(targetRow - ind, targetColumn + ind, piece.Position))
                            break;
                    }
            if (!(targetRow - sourceRow < 0 && targetColumn - sourceColumn < 0))
                for (int ind = 1; ind < Size; ind++)
                    if (CanBeatPiece(newPiece, targetRow + ind, targetColumn + ind, piece))
                    {
                        BeatedPiece beatedPiece = new BeatedPiece(targetRow + ind, targetColumn + ind, piece.Color == PieceColor.White ? PieceColor.Black : PieceColor.White, false, targetRow, targetColumn);
                        if (beatedPieces.Where(p => p.Position == beatedPiece.Position && p.Color == beatedPiece.Color).Count() == 0)
                        {
                            List<BeatedPiece> newBeatedPieces = new List<BeatedPiece>(beatedPieces);
                            newBeatedPieces.Add(beatedPiece);
                            GetAllKingBeatMoves(piece, newBeatedPieces, targetRow, targetColumn, targetRow + ind + 1, targetColumn + ind + 1, ref allMoves, ref maxBeated);
                            break;
                        }
                        else
                            break;
                    }
                    else
                    {
                        if (!CanMoveToPosition(targetRow + ind, targetColumn + ind, piece.Position))
                            break;
                    }
        }

        public override string ToString()
        {
            string boardString = "";
            for (int i = 0; i != Size * Size + 1; i++)
            {
                boardString += $"{i}:{BoardArray[i]} ";
            }
            return boardString;
        }
    }
}
