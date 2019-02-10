/****** Script for SelectTopNRows command from SSMS  ******/


  CREATE VIEW cmn.game_statistics AS
  (
	  SELECT
		g.game_id, 
		g.start_date AS start_date,
		DATEDIFF(SECOND, g.start_date, last_move.end_time) AS seconds_played,
		g.move_count AS move_count,
		g.game_result AS game_result,
		gt.game_type_name AS game_type,
		white_player.player_name AS white_name,
		white_player_algorithm.algorithm_name AS white_algorithm,
		moves_statistics.white_time AS white_avg_move_time_ms,
		black_player.player_name AS black_name,
		black_player_algorithm.algorithm_name AS black_algorithm,
		moves_statistics.black_time AS black_avg_move_time_ms
	  FROM cmn.game g
		JOIN cmn.game_type gt ON gt.game_type_id = g.game_type_id
		JOIN cmn.player_information white_player_information ON white_player_information.player_information_id = g.white_player_information_id
		JOIN cmn.player_information black_player_information ON black_player_information.player_information_id = g.black_player_information_id
		JOIN cmn.algorithm white_player_algorithm ON white_player_algorithm.algorithm_id = white_player_information.algorithm_id
		JOIN cmn.algorithm black_player_algorithm ON black_player_algorithm.algorithm_id = black_player_information.algorithm_id
		JOIN cmn.player white_player ON white_player.player_id = white_player_information.player_id
		JOIN cmn.player black_player ON black_player.player_id = black_player_information.player_id
		CROSS APPLY (
			SELECT TOP 1 end_time
			FROM cmn.game_move gm
			WHERE gm.game_id = g.game_id 
			ORDER BY gm.game_move_id DESC
		) AS last_move
		CROSS APPLY (
			SELECT
				AVG(CASE WHEN player = 'B' THEN DATEDIFF(MILLISECOND, start_time, end_time) ELSE NULL END) black_time,
				AVG(CASE WHEN player = 'W' THEN DATEDIFF(MILLISECOND, start_time, end_time) ELSE NULL END) white_time
			FROM cmn.game_move
			WHERE game_id = g.game_id
		) AS moves_statistics
  )
