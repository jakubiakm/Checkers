using Checkers.Data;
using Checkers.Logic.Engines;
using Checkers.Logic.Enums;
using Checkers.Logic.GameObjects;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace Checkers.Console
{
    class Program
    {
        static void Main(string[] args)
        {
            var game = new Game(new RandomEngine(PieceColor.Black, null), new RandomEngine(PieceColor.White, null), 10, 20, 20, GameVariant.Checkers);
            var random = new Random();
            Stopwatch watch = new Stopwatch();
            watch.Start();
            List<long> times = new List<long>();
            for (int i = 1; i != 10000; i++)
            {
                var board = game.Board;
                var color = PieceColor.White;
                while (true)
                {
                    var moves = board.GetAllPossibleMoves(color);
                    if (moves.Count == 0)
                        break;
                    color = color == PieceColor.Black ? PieceColor.White : PieceColor.Black;
                    board = board.GetBoardAfterMove(moves[random.Next(0, moves.Count - 1)]);
                }
                if (i % 100 == 0)
                {
                    times.Add(watch.ElapsedMilliseconds);
                    System.Console.WriteLine($"{i / 100}\t{times.Last()}\tśrednia: {times.Average()}\tśrednia na symulację: {times.Last() / 100}");
                    watch.Restart();
                }
            }
            System.Console.WriteLine($"Średnia: {times.Average()}");
        }
    }
}
