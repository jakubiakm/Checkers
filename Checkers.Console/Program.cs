using Checkers.Data;
using Checkers.Logic.Engines;
using Checkers.Logic.Enums;
using Checkers.Logic.Exceptions;
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
            SimulateGames();
        }

        class simulator
        {
            public int mctsIterations;
            public int alphaBetaDeep;
            public int numberOfSimulations;
            public simulator(int mctsIt, int alphaBetaDeep, int simulations)
            {
                mctsIterations = mctsIt;
                this.alphaBetaDeep = alphaBetaDeep;
                numberOfSimulations = simulations;
            }
        }

        public static void SimulateGames()
        {
            List<simulator> simulations = new List<simulator>();
            simulations.Add(new simulator(50, 2, 100));
            simulations.Add(new simulator(100, 2, 50));
            simulations.Add(new simulator(150, 2, 33));
            simulations.Add(new simulator(200, 2, 25));

            simulations.Add(new simulator(50, 3, 100));
            simulations.Add(new simulator(100, 3, 50));
            simulations.Add(new simulator(150, 3, 33));
            simulations.Add(new simulator(200, 3, 25));

            simulations.Add(new simulator(50, 4, 100));
            simulations.Add(new simulator(100, 4, 50));
            simulations.Add(new simulator(150, 4, 33));
            simulations.Add(new simulator(200, 4, 25));

            simulations.Add(new simulator(150, 5, 10));
            simulations.Add(new simulator(200, 5, 10));
            simulations.Add(new simulator(250, 5, 10));
            simulations.Add(new simulator(300, 5, 10));
            simulations.Add(new simulator(350, 5, 10));
            simulations.Add(new simulator(400, 5, 10));
            simulations.Add(new simulator(450, 5, 10));
            simulations.Add(new simulator(500, 5, 10));

            simulations.Add(new simulator(200, 6, 10));
            simulations.Add(new simulator(300, 6, 10));
            simulations.Add(new simulator(400, 6, 10));
            simulations.Add(new simulator(500, 6, 10));
            simulations.Add(new simulator(600, 6, 10));
            simulations.Add(new simulator(700, 6, 10));
            simulations.Add(new simulator(800, 6, 10));
            simulations.Add(new simulator(900, 6, 10));

            simulations.Add(new simulator(500, 7, 10));
            simulations.Add(new simulator(7500, 7, 10));
            simulations.Add(new simulator(1000, 7, 10));
            simulations.Add(new simulator(2500, 7, 10));
            simulations.Add(new simulator(5000, 7, 10));
            simulations.Add(new simulator(7500, 7, 10));
            simulations.Add(new simulator(10000, 7, 10));

            simulations.Add(new simulator(500, 8, 10));
            simulations.Add(new simulator(7500, 8, 10));
            simulations.Add(new simulator(1000, 8, 10));
            simulations.Add(new simulator(2500, 8, 10));
            simulations.Add(new simulator(5000, 8, 10));
            simulations.Add(new simulator(7500, 8, 10));
            simulations.Add(new simulator(10000, 8, 10));

            simulations.Add(new simulator(500, 9, 10));
            simulations.Add(new simulator(1000, 9, 10));
            simulations.Add(new simulator(2500, 9, 10));
            simulations.Add(new simulator(5000, 9, 10));
            simulations.Add(new simulator(10000, 9, 10));
            simulations.Add(new simulator(15000, 9, 10));
            simulations.Add(new simulator(20000, 9, 10));
            simulations.Add(new simulator(25000, 9, 10));

            simulations.Add(new simulator(1000, 10, 10));
            simulations.Add(new simulator(5000, 10, 10));
            simulations.Add(new simulator(10000, 10, 10));
            simulations.Add(new simulator(15000, 10, 10));
            simulations.Add(new simulator(20000, 10, 10));
            simulations.Add(new simulator(25000, 10, 10));

            double mctsIterations = 0;
            double mctsUctParameter = 1.5;

            double alphaBetaDeep = 0;
            IEngine white, black;
            foreach (var simulation in simulations)
            {

                mctsIterations = simulation.mctsIterations;
                alphaBetaDeep = simulation.alphaBetaDeep;
                white = new MctsEngine(PieceColor.White, null, mctsUctParameter, (int)mctsIterations);
                black = new AlphaBetaEngine(PieceColor.Black, (int)alphaBetaDeep);

                System.Console.WriteLine($"Gracz biały: MCTS. Liczba iteracji: {mctsIterations},\t parametr UCT: {mctsUctParameter}.");
                System.Console.WriteLine($"Gracz czarny: Alpha-Beta. Głębokość drzewa {alphaBetaDeep}.");


                System.Console.WriteLine($"Liczba iteracji: {simulation.numberOfSimulations}");

                for (int i = 0; i < simulation.numberOfSimulations; i++)
                {
                    Stopwatch sw = new Stopwatch();
                    sw.Start();
                    System.Console.Write($"{i}.\t");
                    SimulateGame(white, black);
                    System.Console.WriteLine();
                    System.Console.WriteLine($"\tczas: {sw.Elapsed.Seconds}s. ");
                }
            }
        }


        public void TestGetMovesTime()
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


        private static void SimulateGame(IEngine white, IEngine black, int boardSize = 10, int numberOfWhitePieces = 20, int numberOfBlackPieces = 20)
        {
            string winner = "";
            int it = 0;
            Game game = new Game(white, black, boardSize, numberOfWhitePieces, numberOfBlackPieces, GameVariant.Checkers);
            try
            {
                while (1 == 1)
                {
                    it++;
                    game.MakeMove(PieceColor.White);
                    game.MakeMove(PieceColor.Black);
                    System.Console.Write($"[{it}]");
                }

            }
            catch (NotAvailableMoveException exception)
            {
                winner = "";
                winner = exception.Color == PieceColor.Black ? "W" : "B";
            }
            catch (NoAvailablePiecesException exception)
            {
                winner = "";
                switch (exception.Color)
                {
                    case PieceColor.White when game.Variant == GameVariant.Checkers:
                    case PieceColor.Black when game.Variant == GameVariant.Anticheckers:
                        winner = "B";
                        break;
                    case PieceColor.Black when game.Variant == GameVariant.Checkers:
                    case PieceColor.White when game.Variant == GameVariant.Anticheckers:
                        winner = "W";
                        break;
                }
            }
            catch (DrawException)
            {
                winner = "D";
            }
            System.Console.WriteLine();
            System.Console.Write($"wygrywa gracz [{winner}] po {it} ruchach.");
        }

    }
}
