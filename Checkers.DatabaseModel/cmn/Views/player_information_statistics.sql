
/****** Script for SelectTopNRows command from SSMS  ******/


CREATE VIEW [cmn].[player_information_statistics] AS
(
	SELECT
		[pi].player_information_id,
		p.player_name,
		a.algorithm_name,
		[pi].number_of_pieces,
		[pi].tree_depth,
		white_stats.games_count + black_stats.games_count AS total_games,
		ISNULL(white_stats.wins, 0) + ISNULL(black_stats.wins, 0) AS total_wins,
		(ISNULL(white_stats.wins, 0) + ISNULL(black_stats.wins, 0)) / CAST((white_stats.games_count + black_stats.games_count) AS real) AS total_win_ratio,
		white_stats.games_count AS total_games_as_white,
		ISNULL(white_stats.wins, 0) AS total_wins_as_white,
		ISNULL(white_stats.wins, 0) / CAST(CASE WHEN white_stats.games_count = 0 THEN 1 ELSE white_stats.games_count END AS real) AS total_win_ratio_as_white,
		black_stats.games_count AS total_games_as_black,
		ISNULL(black_stats.wins, 0) AS total_wins_as_black,
		ISNULL(black_stats.wins, 0) / CAST(CASE WHEN black_stats.games_count = 0 THEN 1 ELSE black_stats.games_count END AS real) AS total_win_ratio_as_black
	FROM cmn.player_information [pi]
		JOIN cmn.player p ON p.player_id = [pi].player_id
		JOIN cmn.algorithm a ON a.algorithm_id = [pi].algorithm_id
		CROSS APPLY (
			SELECT
				COUNT(1) games_count,
				SUM(CASE WHEN g.game_result = 'W' THEN 1 ELSE 0 END) AS wins
			FROM cmn.game g
			WHERE g.white_player_information_id = [pi].player_information_id
		) AS white_stats
		CROSS APPLY (
			SELECT
				COUNT(1) games_count,
				SUM(CASE WHEN g.game_result = 'B' THEN 1 ELSE 0 END) AS wins
			FROM cmn.game g
			WHERE g.black_player_information_id = [pi].player_information_id
		) AS black_stats
)