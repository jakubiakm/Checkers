using Checkers.Logic.GameObjects;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Tests
{
    [TestFixture]
    public class PiecePositionTests
    {
        [Test]
        public void ToPositionTests()
        {
            Assert.AreEqual(1, Piece.ToPosition(9, 1));
            Assert.AreEqual(2, Piece.ToPosition(9, 3));
            Assert.AreEqual(3, Piece.ToPosition(9, 5));
            Assert.AreEqual(4, Piece.ToPosition(9, 7));
            Assert.AreEqual(5, Piece.ToPosition(9, 9));
            Assert.AreEqual(6, Piece.ToPosition(8, 0));
            Assert.AreEqual(7, Piece.ToPosition(8, 2));
            Assert.AreEqual(8, Piece.ToPosition(8, 4));
            Assert.AreEqual(9, Piece.ToPosition(8, 6));
            Assert.AreEqual(10, Piece.ToPosition(8, 8));
            Assert.AreEqual(11, Piece.ToPosition(7, 1));
            Assert.AreEqual(12, Piece.ToPosition(7, 3));
            Assert.AreEqual(13, Piece.ToPosition(7, 5));
            Assert.AreEqual(14, Piece.ToPosition(7, 7));
            Assert.AreEqual(15, Piece.ToPosition(7, 9));
            Assert.AreEqual(16, Piece.ToPosition(6, 0));
            Assert.AreEqual(17, Piece.ToPosition(6, 2));
            Assert.AreEqual(18, Piece.ToPosition(6, 4));
            Assert.AreEqual(19, Piece.ToPosition(6, 6));
            Assert.AreEqual(20, Piece.ToPosition(6, 8));
            Assert.AreEqual(21, Piece.ToPosition(5, 1));
            Assert.AreEqual(22, Piece.ToPosition(5, 3));
            Assert.AreEqual(23, Piece.ToPosition(5, 5));
            Assert.AreEqual(24, Piece.ToPosition(5, 7));
            Assert.AreEqual(25, Piece.ToPosition(5, 9));
            Assert.AreEqual(26, Piece.ToPosition(4, 0));
            Assert.AreEqual(27, Piece.ToPosition(4, 2));
            Assert.AreEqual(28, Piece.ToPosition(4, 4));
            Assert.AreEqual(29, Piece.ToPosition(4, 6));
            Assert.AreEqual(30, Piece.ToPosition(4, 8));
            Assert.AreEqual(31, Piece.ToPosition(3, 1));
            Assert.AreEqual(32, Piece.ToPosition(3, 3));
            Assert.AreEqual(33, Piece.ToPosition(3, 5));
            Assert.AreEqual(34, Piece.ToPosition(3, 7));
            Assert.AreEqual(35, Piece.ToPosition(3, 9));
            Assert.AreEqual(36, Piece.ToPosition(2, 0));
            Assert.AreEqual(37, Piece.ToPosition(2, 2));
            Assert.AreEqual(38, Piece.ToPosition(2, 4));
            Assert.AreEqual(39, Piece.ToPosition(2, 6));
            Assert.AreEqual(40, Piece.ToPosition(2, 8));
            Assert.AreEqual(41, Piece.ToPosition(1, 1));
            Assert.AreEqual(42, Piece.ToPosition(1, 3));
            Assert.AreEqual(43, Piece.ToPosition(1, 5));
            Assert.AreEqual(44, Piece.ToPosition(1, 7));
            Assert.AreEqual(45, Piece.ToPosition(1, 9));
            Assert.AreEqual(46, Piece.ToPosition(0, 0));
            Assert.AreEqual(47, Piece.ToPosition(0, 2));
            Assert.AreEqual(48, Piece.ToPosition(0, 4));
            Assert.AreEqual(49, Piece.ToPosition(0, 6));
            Assert.AreEqual(50, Piece.ToPosition(0, 8));

            Assert.AreEqual(-1, Piece.ToPosition(8, 1));
            Assert.AreEqual(-1, Piece.ToPosition(8, 3));
            Assert.AreEqual(-1, Piece.ToPosition(8, 5));
            Assert.AreEqual(-1, Piece.ToPosition(8, 7));
            Assert.AreEqual(-1, Piece.ToPosition(8, 9));
            Assert.AreEqual(-1, Piece.ToPosition(9, 0));
            Assert.AreEqual(-1, Piece.ToPosition(9, 2));
            Assert.AreEqual(-1, Piece.ToPosition(9, 4));
            Assert.AreEqual(-1, Piece.ToPosition(9, 6));
            Assert.AreEqual(-1, Piece.ToPosition(9, 8));
            Assert.AreEqual(-1, Piece.ToPosition(6, 1));
            Assert.AreEqual(-1, Piece.ToPosition(6, 3));
            Assert.AreEqual(-1, Piece.ToPosition(6, 5));
            Assert.AreEqual(-1, Piece.ToPosition(6, 7));
            Assert.AreEqual(-1, Piece.ToPosition(6, 9));
            Assert.AreEqual(-1, Piece.ToPosition(7, 0));
            Assert.AreEqual(-1, Piece.ToPosition(7, 2));
            Assert.AreEqual(-1, Piece.ToPosition(7, 4));
            Assert.AreEqual(-1, Piece.ToPosition(7, 6));
            Assert.AreEqual(-1, Piece.ToPosition(7, 8));
            Assert.AreEqual(-1, Piece.ToPosition(4, 1));
            Assert.AreEqual(-1, Piece.ToPosition(4, 3));
            Assert.AreEqual(-1, Piece.ToPosition(4, 5));
            Assert.AreEqual(-1, Piece.ToPosition(4, 7));
            Assert.AreEqual(-1, Piece.ToPosition(4, 9));
            Assert.AreEqual(-1, Piece.ToPosition(5, 0));
            Assert.AreEqual(-1, Piece.ToPosition(5, 2));
            Assert.AreEqual(-1, Piece.ToPosition(5, 4));
            Assert.AreEqual(-1, Piece.ToPosition(5, 6));
            Assert.AreEqual(-1, Piece.ToPosition(5, 8));
            Assert.AreEqual(-1, Piece.ToPosition(2, 1));
            Assert.AreEqual(-1, Piece.ToPosition(2, 3));
            Assert.AreEqual(-1, Piece.ToPosition(2, 5));
            Assert.AreEqual(-1, Piece.ToPosition(2, 7));
            Assert.AreEqual(-1, Piece.ToPosition(2, 9));
            Assert.AreEqual(-1, Piece.ToPosition(3, 0));
            Assert.AreEqual(-1, Piece.ToPosition(3, 2));
            Assert.AreEqual(-1, Piece.ToPosition(3, 4));
            Assert.AreEqual(-1, Piece.ToPosition(3, 6));
            Assert.AreEqual(-1, Piece.ToPosition(3, 8));
            Assert.AreEqual(-1, Piece.ToPosition(0, 1));
            Assert.AreEqual(-1, Piece.ToPosition(0, 3));
            Assert.AreEqual(-1, Piece.ToPosition(0, 5));
            Assert.AreEqual(-1, Piece.ToPosition(0, 7));
            Assert.AreEqual(-1, Piece.ToPosition(0, 9));
            Assert.AreEqual(-1, Piece.ToPosition(1, 0));
            Assert.AreEqual(-1, Piece.ToPosition(1, 2));
            Assert.AreEqual(-1, Piece.ToPosition(1, 4));
            Assert.AreEqual(-1, Piece.ToPosition(1, 6));
            Assert.AreEqual(-1, Piece.ToPosition(1, 8));
        }

        [Test]
        public void ToRowTests()
        {
            Assert.AreEqual(9, Piece.ToRow(1, 10));
            Assert.AreEqual(9, Piece.ToRow(2, 10));
            Assert.AreEqual(9, Piece.ToRow(3, 10));
            Assert.AreEqual(9, Piece.ToRow(4, 10));
            Assert.AreEqual(9, Piece.ToRow(5, 10));
            Assert.AreEqual(8, Piece.ToRow(6, 10));
            Assert.AreEqual(8, Piece.ToRow(7, 10));
            Assert.AreEqual(8, Piece.ToRow(8, 10));
            Assert.AreEqual(8, Piece.ToRow(9, 10));
            Assert.AreEqual(8, Piece.ToRow(10, 10));
            Assert.AreEqual(7, Piece.ToRow(11, 10));
            Assert.AreEqual(7, Piece.ToRow(12, 10));
            Assert.AreEqual(7, Piece.ToRow(13, 10));
            Assert.AreEqual(7, Piece.ToRow(14, 10));
            Assert.AreEqual(7, Piece.ToRow(15, 10));
            Assert.AreEqual(6, Piece.ToRow(16, 10));
            Assert.AreEqual(6, Piece.ToRow(17, 10));
            Assert.AreEqual(6, Piece.ToRow(18, 10));
            Assert.AreEqual(6, Piece.ToRow(19, 10));
            Assert.AreEqual(6, Piece.ToRow(20, 10));
            Assert.AreEqual(5, Piece.ToRow(21, 10));
            Assert.AreEqual(5, Piece.ToRow(22, 10));
            Assert.AreEqual(5, Piece.ToRow(23, 10));
            Assert.AreEqual(5, Piece.ToRow(24, 10));
            Assert.AreEqual(5, Piece.ToRow(25, 10));
            Assert.AreEqual(4, Piece.ToRow(26, 10));
            Assert.AreEqual(4, Piece.ToRow(27, 10));
            Assert.AreEqual(4, Piece.ToRow(28, 10));
            Assert.AreEqual(4, Piece.ToRow(29, 10));
            Assert.AreEqual(4, Piece.ToRow(30, 10));
            Assert.AreEqual(3, Piece.ToRow(31, 10));
            Assert.AreEqual(3, Piece.ToRow(32, 10));
            Assert.AreEqual(3, Piece.ToRow(33, 10));
            Assert.AreEqual(3, Piece.ToRow(34, 10));
            Assert.AreEqual(3, Piece.ToRow(35, 10));
            Assert.AreEqual(2, Piece.ToRow(36, 10));
            Assert.AreEqual(2, Piece.ToRow(37, 10));
            Assert.AreEqual(2, Piece.ToRow(38, 10));
            Assert.AreEqual(2, Piece.ToRow(39, 10));
            Assert.AreEqual(2, Piece.ToRow(40, 10));
            Assert.AreEqual(1, Piece.ToRow(41, 10));
            Assert.AreEqual(1, Piece.ToRow(42, 10));
            Assert.AreEqual(1, Piece.ToRow(43, 10));
            Assert.AreEqual(1, Piece.ToRow(44, 10));
            Assert.AreEqual(1, Piece.ToRow(45, 10));
            Assert.AreEqual(0, Piece.ToRow(46, 10));
            Assert.AreEqual(0, Piece.ToRow(47, 10));
            Assert.AreEqual(0, Piece.ToRow(48, 10));
            Assert.AreEqual(0, Piece.ToRow(49, 10));
            Assert.AreEqual(0, Piece.ToRow(50, 10));
        }

        [Test]
        public void ToColumnTests()
        {
            Assert.AreEqual(1, Piece.ToColumn(1, 10));
            Assert.AreEqual(3, Piece.ToColumn(2, 10));
            Assert.AreEqual(5, Piece.ToColumn(3, 10));
            Assert.AreEqual(7, Piece.ToColumn(4, 10));
            Assert.AreEqual(9, Piece.ToColumn(5, 10));
            Assert.AreEqual(0, Piece.ToColumn(6, 10));
            Assert.AreEqual(2, Piece.ToColumn(7, 10));
            Assert.AreEqual(4, Piece.ToColumn(8, 10));
            Assert.AreEqual(6, Piece.ToColumn(9, 10));
            Assert.AreEqual(8, Piece.ToColumn(10, 10));
            Assert.AreEqual(1, Piece.ToColumn(11, 10));
            Assert.AreEqual(3, Piece.ToColumn(12, 10));
            Assert.AreEqual(5, Piece.ToColumn(13, 10));
            Assert.AreEqual(7, Piece.ToColumn(14, 10));
            Assert.AreEqual(9, Piece.ToColumn(15, 10));
            Assert.AreEqual(0, Piece.ToColumn(16, 10));
            Assert.AreEqual(2, Piece.ToColumn(17, 10));
            Assert.AreEqual(4, Piece.ToColumn(18, 10));
            Assert.AreEqual(6, Piece.ToColumn(19, 10));
            Assert.AreEqual(8, Piece.ToColumn(20, 10));
            Assert.AreEqual(1, Piece.ToColumn(21, 10));
            Assert.AreEqual(3, Piece.ToColumn(22, 10));
            Assert.AreEqual(5, Piece.ToColumn(23, 10));
            Assert.AreEqual(7, Piece.ToColumn(24, 10));
            Assert.AreEqual(9, Piece.ToColumn(25, 10));
            Assert.AreEqual(0, Piece.ToColumn(26, 10));
            Assert.AreEqual(2, Piece.ToColumn(27, 10));
            Assert.AreEqual(4, Piece.ToColumn(28, 10));
            Assert.AreEqual(6, Piece.ToColumn(29, 10));
            Assert.AreEqual(8, Piece.ToColumn(30, 10));
            Assert.AreEqual(1, Piece.ToColumn(31, 10));
            Assert.AreEqual(3, Piece.ToColumn(32, 10));
            Assert.AreEqual(5, Piece.ToColumn(33, 10));
            Assert.AreEqual(7, Piece.ToColumn(34, 10));
            Assert.AreEqual(9, Piece.ToColumn(35, 10));
            Assert.AreEqual(0, Piece.ToColumn(36, 10));
            Assert.AreEqual(2, Piece.ToColumn(37, 10));
            Assert.AreEqual(4, Piece.ToColumn(38, 10));
            Assert.AreEqual(6, Piece.ToColumn(39, 10));
            Assert.AreEqual(8, Piece.ToColumn(40, 10));
            Assert.AreEqual(1, Piece.ToColumn(41, 10));
            Assert.AreEqual(3, Piece.ToColumn(42, 10));
            Assert.AreEqual(5, Piece.ToColumn(43, 10));
            Assert.AreEqual(7, Piece.ToColumn(44, 10));
            Assert.AreEqual(9, Piece.ToColumn(45, 10));
            Assert.AreEqual(0, Piece.ToColumn(46, 10));
            Assert.AreEqual(2, Piece.ToColumn(47, 10));
            Assert.AreEqual(4, Piece.ToColumn(48, 10));
            Assert.AreEqual(6, Piece.ToColumn(49, 10));
            Assert.AreEqual(8, Piece.ToColumn(50, 10));
        }
    }
}
