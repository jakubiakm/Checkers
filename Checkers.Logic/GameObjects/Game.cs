using Checkers.Data;
using Checkers.Logic.Engines;
using Checkers.Logic.Enums;
using Checkers.Logic.Exceptions;
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.GameObjects
{
    public class Game
    {
        private DatabaseLayer _databaseLayer = new DatabaseLayer();

        public List<HistoryBoard> History { get; private set; }

        public CheckersBoard Board { get; private set; }

        public IEngine WhitePlayerEngine { get; private set; }

        public IEngine BlackPlayerEngine { get; private set; }

        public GameVariant Variant { get; set; }

        public DateTime StartDate { get; set; }

        public List<Move> GameMoves { get; set; }

        public Game(IEngine whiteEngine, IEngine blackEngine, int boardSize, int numberOfWhitePieces, int numberOfBlackPieces, GameVariant variant)
        {
            WhitePlayerEngine = whiteEngine;
            BlackPlayerEngine = blackEngine;
            Board = new CheckersBoard(boardSize, numberOfWhitePieces, numberOfBlackPieces);
            GameMoves = new List<Move>();
            History = new List<HistoryBoard>();
            Variant = variant;
            StartDate = DateTime.Now;
        }

        public Game(IEngine whiteEngine, IEngine blackEngine, int boardSize, List<Piece> pieces, GameVariant variant)
        {
            WhitePlayerEngine = whiteEngine;
            BlackPlayerEngine = blackEngine;
            Board = new CheckersBoard(boardSize, pieces);
            History = new List<HistoryBoard>();
            Variant = variant;
            StartDate = DateTime.Now;
            GameMoves = new List<Move>();
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
                        Board.LastMove = Board.MakeMove(WhitePlayerEngine.MakeMove(Board, Variant, GameMoves));
                        break;
                    case PieceColor.Black:
                        Board.LastMove = Board.MakeMove(BlackPlayerEngine.MakeMove(Board, Variant, GameMoves));
                        break;
                }
                var endTime = DateTime.Now;
                History.Add(new HistoryBoard(startTime, endTime, board, color));
                if (Board.BoardArray.Where(p => p < 0).Count() == 0)
                {
                    throw new NoAvailablePiecesException(PieceColor.Black, Board.LastMove);
                }
                if (Board.BoardArray.Where(p => p > 0).Count() == 0)
                {
                    throw new NoAvailablePiecesException(PieceColor.White, Board.LastMove);
                }
                GameMoves.Add(Board.LastMove);
                if (IsDraw())
                {
                    throw new DrawException();
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
                switch (exception.Color)
                {
                    case PieceColor.White when Variant == GameVariant.Checkers:
                    case PieceColor.Black when Variant == GameVariant.Anticheckers:
                        winner = "B";
                        break;
                    case PieceColor.Black when Variant == GameVariant.Checkers:
                    case PieceColor.White when Variant == GameVariant.Anticheckers:
                        winner = "W";
                        break;
                }
                AddGameToDatabase(winner);
                throw;
            }
            catch(DrawException)
            {
                AddGameToDatabase("D");
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
            if (WhitePlayerEngine.GetType() == typeof(AlphaBetaEngine))
            {
                var engine = (AlphaBetaEngine)WhitePlayerEngine;
                whitePlayerInformation.tree_depth = engine.AlphaBetaTreeDepth;
            }
            if (BlackPlayerEngine.GetType() == typeof(AlphaBetaEngine))
            {
                var engine = (AlphaBetaEngine)BlackPlayerEngine;
                blackPlayerInformation.tree_depth = engine.AlphaBetaTreeDepth;
            }
            if (WhitePlayerEngine.GetType() == typeof(MctsEngine))
            {
                var engine = (MctsEngine)WhitePlayerEngine;
                whitePlayerInformation.uct_parameter = engine.UctParameter;
                whitePlayerInformation.number_of_iterations = engine.NumberOfIterations;
            }
            if (BlackPlayerEngine.GetType() == typeof(MctsEngine))
            {
                var engine = (MctsEngine)BlackPlayerEngine;
                blackPlayerInformation.uct_parameter = engine.UctParameter;
                blackPlayerInformation.number_of_iterations = engine.NumberOfIterations;
            }
            game_type gameType = new game_type() { game_type_name = Variant.ToString() };
            List<game_move> gameMoves = new List<game_move>();
            foreach (var move in History.Skip(1))
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
            int moveCount = History.Count;
            _databaseLayer.AddGame(whitePlayerInformation, blackPlayerInformation, gameType, gameMoves, Board.Size, winner, moveCount, StartDate);
        }


        private bool IsDraw()
        {
            //Jeżeli przez 10 kolejnych posunięć obu graczy, jedynie damki były przestawiane, nie wykonano żadnego ruchu pionem i nie wykonano żadnego bicia to grę uważa się za remisową.
            var lastMoves = GameMoves.Skip(Math.Max(0, GameMoves.Count - 2 * 10)).ToList();
            foreach(var move in lastMoves)
            {
                if (move.BeatedPieces != null && move.BeatedPieces.Count > 0)
                    return false;
                if (!move.OldPiece.IsKing)
                    return false;
            }

            return true;
        }
    }
}
