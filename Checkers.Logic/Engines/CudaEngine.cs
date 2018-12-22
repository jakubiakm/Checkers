﻿using Checkers.Logic.Enums;
using Checkers.Logic.Exceptions;
using Checkers.Logic.GameObjects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.Engines
{
    public class CudaEngine : IEngine
    {
        public PieceColor Color { get; set; }

        public CudaEngine(PieceColor color)
        {
            Color = color;
        }

        [DllImport(@"D:\Users\syntaximus\Documents\GitHub\Checkers\x64\Debug\Checkers.Cuda.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall)]
        public static extern int MakeMoveGpu(int size, int player, int[] board, int[] possibleMoves);

        public Move MakeMove(CheckersBoard currentBoard)
        {
            Random random = new Random();
            List<Move> allPossibleMoves = currentBoard.GetAllPossibleMoves(Color);
            int count = allPossibleMoves.Count;
            if (count == 0)
                throw new NotAvailableMoveException(Color);
            int elemIndex = MakeMoveGpu(currentBoard.Size, Color == PieceColor.White ? 0 : 1, currentBoard.GetBoardArray(), GetPossibleMovesArray(allPossibleMoves));
            return allPossibleMoves[elemIndex];
        }

        public int[] GetPossibleMovesArray(List<Move> allPossibleMoves)
        {
            List<int> possibleMoves = new List<int>();
            possibleMoves.Add(allPossibleMoves.Count);
            foreach (var move in allPossibleMoves)
            {
                possibleMoves.Add(move.BeatedPieces?.Count ?? 0);
                for (int i = 0; i != (move.BeatedPieces?.Count ?? 0); i++)
                {
                    possibleMoves.Add(move.BeatedPieces[i].Position);
                }
                possibleMoves.Add(move.OldPiece.Position);
                possibleMoves.Add(move.NewPiece.Position);
            }
            return possibleMoves.ToArray();
        }
    }
}