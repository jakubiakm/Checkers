
  
  
/****** Script for SelectTopNRows command from SSMS  ******/  
  
  
CREATE VIEW [cmn].[player_information_statistics] AS  
	WITH stats AS (
		SELECT TOP 10000
			  white_player_information_id 
			, black_player_information_id 
			, SUM(CASE WHEN g.game_result = 'W' THEN 1 ELSE 0 END) AS white_wins
			, SUM(CASE WHEN g.game_result = 'B' THEN 1 ELSE 0 END) AS black_wins
			, SUM(CASE WHEN g.game_result = 'D' THEN 1 ELSE 0 END) AS draws
			, CAST(SUM(1) AS real) AS games_number
			, AVG(moves_statistics.white_time) AS white_time
			, AVG(moves_statistics.black_time) AS black_time
		FROM 
			cmn.game g
			CROSS APPLY (
				SELECT 
					AVG(CASE 
							WHEN player = 'B' THEN NULL 
							ELSE CASE 
								WHEN DATEDIFF(MILLISECOND, start_time, end_time) = 0 THEN NULL 
								ELSE DATEDIFF(MILLISECOND, start_time, end_time) 
							END 
						END) AS white_time,
					AVG(CASE 
							WHEN player = 'W' THEN NULL 
							ELSE 
								CASE WHEN DATEDIFF(MILLISECOND, start_time, end_time) = 0 THEN NULL 
								ELSE DATEDIFF(MILLISECOND, start_time, end_time) 
							END 
						END) AS black_time
				FROM cmn.game_move WHERE game_id = g.game_id
			) AS moves_statistics
		GROUP BY 
			g.white_player_information_id, 
			g.black_player_information_id
		ORDER BY 
			MAX(g.start_date) DESC
	)
	SELECT
		CONCAT(
			wa.algorithm_name,
			' ',
			CASE 
				WHEN wpi.number_of_iterations IS NOT NULL THEN CONCAT('iteracje: ', wpi.number_of_iterations) 
				WHEN wpi.tree_depth IS NOT NULL THEN CONCAT('głębokość: ', wpi.tree_depth) 
			END
		) AS white_player
		, CONCAT(
			ba.algorithm_name,
			' ',
			CASE 
				WHEN bpi.number_of_iterations IS NOT NULL THEN CONCAT('iteracje: ', bpi.number_of_iterations) 
				WHEN bpi.tree_depth IS NOT NULL THEN CONCAT('głębokość: ', bpi.tree_depth) 
			END
		) AS black_player
		, s.games_number
		, s.white_wins
		, s.black_wins
		, s.draws
		, s.white_wins / s.games_number AS white_wins_ratio
		, s.black_wins / s.games_number AS black_wins_ratio
		, s.draws / s.games_number AS draws_ratio
		, s.white_time AS white_move_average_time
		, s.black_time AS black_move_average_time
	FROM 
		stats s
		JOIN cmn.player_information wpi ON wpi.player_information_id = s.white_player_information_id
		JOIN cmn.player_information bpi ON bpi.player_information_id = s.black_player_information_id
		JOIN cmn.algorithm wa ON wa.algorithm_id = wpi.algorithm_id
		JOIN cmn.algorithm ba ON ba.algorithm_id = bpi.algorithm_id