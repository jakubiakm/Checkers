using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Data
{
    public class DatabaseLayer
    {
        private CheckersEntities context = new CheckersEntities();

        public void AddGame(player_information whitePlayerInformation, player_information blackPlayerInformation, game_type gameType, List<game_move> gameMoves, int gameSize, string gameResult, int moveCount, DateTime startDate)
        {
            game game = new game();

            whitePlayerInformation.player = AddOrLoadPlayer(whitePlayerInformation.player);
            whitePlayerInformation.algorithm = AddOrLoadAlgorithm(whitePlayerInformation.algorithm);
            whitePlayerInformation = AddOrLoadPlayerInformation(whitePlayerInformation);

            context.SaveChanges();

            blackPlayerInformation.player = AddOrLoadPlayer(blackPlayerInformation.player);
            blackPlayerInformation.algorithm = AddOrLoadAlgorithm(blackPlayerInformation.algorithm);
            blackPlayerInformation = AddOrLoadPlayerInformation(blackPlayerInformation);

            context.SaveChanges();

            game.white_player_information_id = whitePlayerInformation.player_information_id;
            game.black_player_information_id = blackPlayerInformation.player_information_id;
            game.game_type = AddOrLoadGameType(gameType);
            game.game_size = gameSize;
            game.game_result = gameResult;
            game.move_count = moveCount;
            game.start_date = startDate;

            game = context.games.Add(game);
            
            foreach(var game_move in gameMoves)
            {
                game_move.game_id = game.game_id;
                context.game_move.Add(game_move);
            }

            context.SaveChanges();
        }

        private player AddOrLoadPlayer(player player)
        {
            if (!context.players.Any(p => p.player_name == player.player_name))
            {
                player = context.players.Add(player);
            }
            else
            {
                player = context.players.First(p => p.player_name == player.player_name);
            }
            return player;
        }

        private algorithm AddOrLoadAlgorithm(algorithm algorithm)
        {
            if (!context.algorithms.Any(p => p.algorithm_name == algorithm.algorithm_name))
            {
                algorithm = context.algorithms.Add(algorithm);
            }
            else
            {
                algorithm = context.algorithms.First(algo => algo.algorithm_name == algorithm.algorithm_name);
            }
            return algorithm;
        }

        private player_information AddOrLoadPlayerInformation(player_information playerInformation)
        {
            if (!context.player_information.Any(p => 
                p.player_id == playerInformation.player.player_id &&
                p.algorithm_id == playerInformation.algorithm.algorithm_id &&
                p.number_of_pieces == playerInformation.number_of_pieces))
            {
                playerInformation = context.player_information.Add(playerInformation);
            }
            else
            {
                playerInformation = context.player_information.First(p =>
                    p.player_id == playerInformation.player.player_id &&
                    p.algorithm_id == playerInformation.algorithm.algorithm_id &&
                    p.number_of_pieces == playerInformation.number_of_pieces);
            }
            return playerInformation;
        }

        private game_type AddOrLoadGameType(game_type gameType)
        {
            if (!context.game_type.Any(p => p.game_type_name == gameType.game_type_name))
            {
                gameType = context.game_type.Add(gameType);
            }
            else
            {
                gameType = context.game_type.First(type => type.game_type_name == type.game_type_name);
            }
            return gameType;
        }
    }
}
