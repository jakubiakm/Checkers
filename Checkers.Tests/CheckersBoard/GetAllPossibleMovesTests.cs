using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Checkers.Logic.GameObjects;
using Checkers.Logic.Enums;

namespace Checkers.Tests
{
    [TestFixture]
    public class GetAllPossibleMovesTests
    {
        [Test]
        public void Test1()
        {
            CheckersBoard board = new CheckersBoard(10, new List<Piece>()
            {
                new Piece(4,4, PieceColor.White, true),
                new Piece(6,6, PieceColor.Black, false),
                new Piece(7,7, PieceColor.Black, false),
                new Piece(8,8, PieceColor.Black, false)
            });
            var moves = board.GetAllPossibleMoves(PieceColor.White);
            Assert.IsTrue(moves.Where(m => m.BeatedPieces == null).Count() > 0);
        }

        [Test]
        public void Test2()
        {
            CheckersBoard board = new CheckersBoard(10, new List<Piece>()
            {
                new Piece(1,1, PieceColor.White, true),
                new Piece(5,5, PieceColor.Black, false),
                new Piece(7,7, PieceColor.Black, false)
            });
            var moves = board.GetAllPossibleMoves(PieceColor.White);
            Assert.IsTrue(moves.Where(m => m.BeatedPieces?.Count == 2).Count() > 0);
        }

        [Test]
        public void Test3()
        {
            CheckersBoard board = new CheckersBoard(10, new List<Piece>()
            {
                new Piece(1,1, PieceColor.White, true),
                new Piece(5,5, PieceColor.Black, false),
                new Piece(8,8, PieceColor.Black, false)
            });
            var moves = board.GetAllPossibleMoves(PieceColor.White);
            Assert.IsTrue(moves.Where(m => m.BeatedPieces?.Count == 2).Count() > 0);
        }

        [Test]
        public void Test4()
        {
            CheckersBoard board = new CheckersBoard(10, new List<Piece>()
            {
                new Piece(3,3, PieceColor.White, true),
                new Piece(1,1, PieceColor.Black, false),
                new Piece(8,8, PieceColor.Black, false)
            });
            var moves = board.GetAllPossibleMoves(PieceColor.White);
            Assert.IsTrue(moves.Where(m => m.BeatedPieces?.Count == 1).Count() == 2);
        }

        [Test]
        public void Test5()
        {
            CheckersBoard board = new CheckersBoard(10, new List<Piece>()
            {
                new Piece(1,1, PieceColor.White, true),
                new Piece(3,3, PieceColor.Black, false),
                new Piece(2,4, PieceColor.Black, false),
                new Piece(2,6, PieceColor.Black, false),
                new Piece(2,8, PieceColor.Black, false),
                new Piece(4,8, PieceColor.Black, false),
                new Piece(8,2, PieceColor.Black, false),
                new Piece(8,4, PieceColor.Black, false),
                new Piece(6,6, PieceColor.Black, false)
            });
            var moves = board.GetAllPossibleMoves(PieceColor.White);
            Assert.IsTrue(moves.Where(m => m.BeatedPieces?.Count == 7).Count() > 0 && moves.Where(m => m.BeatedPieces.Last().Row == 4 || m.BeatedPieces.Last().Column == 8).Count() == 0);
        }

        [Test]
        public void Test6()
        {
            CheckersBoard board = new CheckersBoard(10, new List<Piece>()
            {
                new Piece(1,1, PieceColor.White, true),
                new Piece(3,3, PieceColor.Black, false),
                new Piece(2,4, PieceColor.Black, false),
                new Piece(3,5, PieceColor.Black, false),
                new Piece(2,8, PieceColor.Black, false),
                new Piece(1,5, PieceColor.Black, false),
                new Piece(3,7, PieceColor.Black, false),
                new Piece(7,3, PieceColor.Black, false),
                new Piece(7,5, PieceColor.Black, false),
                new Piece(5,7, PieceColor.Black, false),
                new Piece(0,4, PieceColor.Black, false),

            });
            var moves = board.GetAllPossibleMoves(PieceColor.White);
            Assert.IsTrue(moves.Where(m => m.BeatedPieces?.Count == 6).Count() > 0);
        }

        [Test]
        public void Test7()
        {
            CheckersBoard board = new CheckersBoard(10, new List<Piece>()
            {
                new Piece(1,1, PieceColor.White, false),
                new Piece(3,3, PieceColor.Black, false),
                new Piece(2,4, PieceColor.Black, false),
                new Piece(3,5, PieceColor.Black, false),
                new Piece(2,8, PieceColor.Black, false),
                new Piece(1,5, PieceColor.Black, false),
                new Piece(3,7, PieceColor.Black, false),
                new Piece(7,3, PieceColor.Black, false),
                new Piece(7,5, PieceColor.Black, false),
                new Piece(5,7, PieceColor.Black, false)
            });
            var moves = board.GetAllPossibleMoves(PieceColor.White);
            Assert.IsTrue(moves.Where(m => (m.BeatedPieces?.Count ?? 0) == 0).Count() > 0);
        }

        [Test]
        public void Test8()
        {
            CheckersBoard board = new CheckersBoard(10, new List<Piece>()
            {
                new Piece(2,2, PieceColor.White, false),
                new Piece(3,3, PieceColor.Black, false),
                new Piece(2,4, PieceColor.Black, false),
                new Piece(3,5, PieceColor.Black, false),
                new Piece(2,8, PieceColor.Black, false),
                new Piece(1,5, PieceColor.Black, false),
                new Piece(3,7, PieceColor.Black, false),
                new Piece(7,3, PieceColor.Black, false),
                new Piece(7,5, PieceColor.Black, false),
                new Piece(5,7, PieceColor.Black, false)
            });
            var moves = board.GetAllPossibleMoves(PieceColor.White);
            Assert.IsTrue(moves.Where(m => m.BeatedPieces?.Count == 6).Count() > 0);
        }

        [Test]
        public void Test9()
        {
            CheckersBoard board = new CheckersBoard(10, new List<Piece>()
            {
                new Piece(3, 5, PieceColor.White, true),
                new Piece(4, 6, PieceColor.Black, true),
                new Piece(6, 8, PieceColor.Black, true),
                new Piece(7, 5, PieceColor.Black, true),
                new Piece(7, 3, PieceColor.Black, true),
                new Piece(3, 1, PieceColor.Black, true)
            });
            var moves = board.GetAllPossibleMoves(PieceColor.White);
            Assert.IsTrue(moves.Where(m => m.BeatedPieces?.Count == 4).Count() > 0);
        }
    }
}

