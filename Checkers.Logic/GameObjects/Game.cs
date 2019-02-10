using Checkers.Data;
using Checkers.Logic.Engines;
using Checkers.Logic.Enums;
using Checkers.Logic.Exceptions;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.GameObjects
{
    public class Game
    {
        public List<HistoryBoard> History { get; private set; }

        public CheckersBoard Board { get; private set; }

        public IEngine WhitePlayerEngine { get; private set; }

        public IEngine BlackPlayerEngine { get; private set; }

        public GameVariant Variant { get; set; }

        public Game(IEngine whiteEngine, IEngine blackEngine, int boardSize, int numberOfWhitePieces, int numberOfBlackPieces, GameVariant variant)
        {
            WhitePlayerEngine = whiteEngine;
            BlackPlayerEngine = blackEngine;
            Board = new CheckersBoard(boardSize, numberOfWhitePieces, numberOfBlackPieces);
            History = new List<HistoryBoard>();
            Variant = variant;
        }

        public Game(IEngine whiteEngine, IEngine blackEngine, int boardSize, List<Piece> pieces, GameVariant variant)
        {
            WhitePlayerEngine = whiteEngine;
            BlackPlayerEngine = blackEngine;
            Board = new CheckersBoard(boardSize, pieces);
            History = new List<HistoryBoard>();
            Variant = variant;
        }

        public Move MakeMove(PieceColor color)
        {
            try
            {
                var startTime = DateTime.Now;
                var board = Board.DeepClone();
                switch (color)
                {
                    case PieceColor.White:
                        Board.LastMove = Board.MakeMove(WhitePlayerEngine.MakeMove(Board, Variant));
                        break;
                    case PieceColor.Black:
                        Board.LastMove = Board.MakeMove(BlackPlayerEngine.MakeMove(Board, Variant));
                        break;
                }
                var endTime = DateTime.Now;
                History.Add(new HistoryBoard(startTime, endTime, board, color));
                if (Board.PiecesOnBoard.Where(p => p.Color == PieceColor.Black).Count() == 0)
                {
                    throw new NoAvailablePiecesException(PieceColor.Black, Board.LastMove);
                }
                if (Board.PiecesOnBoard.Where(p => p.Color == PieceColor.White).Count() == 0)
                {
                    throw new NoAvailablePiecesException(PieceColor.White, Board.LastMove);
                }
                return Board.LastMove;
            }
            catch (NotAvailableMoveException exception)
            {
                string winner = "";
                winner = exception.Color == PieceColor.Black ? "W" : "B";
                AddGameToDatabase(winner);
                throw;
            }
            catch (NoAvailablePiecesException exception)
            {
                string winner = "";
                winner = exception.Color == PieceColor.Black ? 
                    Variant == GameVariant.Checkers ? "W" : "B" :
                    Variant == GameVariant.Anticheckers ? "B" : "W";
                AddGameToDatabase(winner);
                throw;
            }
            catch
            {
                throw;
            }
        }

        private void AddGameToDatabase(string winner)
        {
            player_information whitePlayerInformation = new player_information()
            {
                algorithm = new algorithm()
                {
                    algorithm_name = WhitePlayerEngine.Kind.ToString()
                },
                number_of_pieces = Board.NumberOfWhitePiecesAtBeggining,
                player = WhitePlayerEngine.Kind == EngineKind.Human ? new player() { player_name = "syntaximus" } : new player() { player_name = "CPU" }
            };
            player_information blackPlayerInformation = new player_information()
            {
                algorithm = new algorithm()
                {
                    algorithm_name = BlackPlayerEngine.Kind.ToString()
                },
                number_of_pieces = Board.NumberOfBlackPiecesAtBeggining,
                player = BlackPlayerEngine.Kind == EngineKind.Human ? new player() { player_name = "syntaximus" } : new player() { player_name = "CPU" }
            };
            game_type gameType = new game_type() { game_type_name = Variant.ToString() };
            List<game_move> gameMoves = new List<game_move>();
            foreach(var move in History.Skip(1))
            {
                var gameMove = new game_move()
                {
                    player = move.PieceColor == PieceColor.White ? "W" : "B",
                    start_time = move.StartTime,
                    end_time = move.EndTime,
                    from_position = move.Board.LastMove.OldPiece.Position,
                    to_position = move.Board.LastMove.NewPiece.Position,
                    beated_pieces_count = move.Board.LastMove.BeatedPieces?.Count ?? 0,
                    beated_pieces = move.Board.LastMove.GetBeatedPiecesString(),
                    board_after_move = move.Board.ToString()
                };
                gameMoves.Add(gameMove);
            }
            DatabaseLayer.AddGame(whitePlayerInformation, blackPlayerInformation, gameType, gameMoves, Board.Size, winner);
        }
    }
}
