using Checkers.Logic.Enums;
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
        public EngineKind Kind
        {
            get
            {
                return EngineKind.Cuda;
            }
        }

        public PieceColor Color { get; set; }

        public int MctsIterationCount { get; set; }

        public int GridSize { get; set; }

        public int BlockSize { get; set; }

        public CudaEngine(PieceColor color, int mctsIterationCount, int gridSize, int blockSize)
        {
            Color = color;
            MctsIterationCount = mctsIterationCount;
            GridSize = gridSize;
            BlockSize = blockSize;
        }

#if DEBUG
        [DllImport(@"D:\Users\syntaximus\Documents\GitHub\Checkers\x64\Debug\Checkers.Cuda.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall)]
        public static extern int MakeMoveGpu(char size, int player, char[] board, char[] possibleMoves, int mctsIterationCount, int gridSize, int blockSize, int gameVariant);
#else
        [DllImport(@"D:\Users\syntaximus\Documents\GitHub\Checkers\x64\Release\Checkers.Cuda.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall)]
        public static extern int MakeMoveGpu(char size, int player, char[] board, char[] possibleMoves, int mctsIterationCount, int gridSize, int blockSize, int gameVariant);
#endif

        public void Reset()
        {

        }

        public Move MakeMove(CheckersBoard currentBoard, GameVariant variant)
        {
            Random random = new Random();
            List<Move> allPossibleMoves = currentBoard.GetAllPossibleMoves(Color);
            int count = allPossibleMoves.Count;
            if (count == 0)
                throw new NotAvailableMoveException(Color);
            int elemIndex = MakeMoveGpu((char)currentBoard.Size, Color == PieceColor.White ? 0 : 1, currentBoard.GetBoardArray(), GetPossibleMovesArray(allPossibleMoves), MctsIterationCount, GridSize, BlockSize, variant == GameVariant.Checkers ? 0 : 1);          return allPossibleMoves[elemIndex];
        }

        public char[] GetPossibleMovesArray(List<Move> allPossibleMoves)
        {
            List<char> possibleMoves = new List<char>();
            possibleMoves.Add((char)allPossibleMoves.Count);
            foreach (var move in allPossibleMoves)
            {
                possibleMoves.Add(move.BeatedPieces?.Count == null ? (char)0 : (char)move.BeatedPieces?.Count);
                for (int i = 0; i != (move.BeatedPieces?.Count ?? 0); i++)
                {
                    possibleMoves.Add((char)move.BeatedPieces[i].Position);
                }
                possibleMoves.Add((char)move.OldPiece.Position);
                possibleMoves.Add((char)move.NewPiece.Position);
            }
            return possibleMoves.ToArray();
        }
    }
}
