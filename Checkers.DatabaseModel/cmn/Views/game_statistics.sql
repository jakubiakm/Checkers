/****** Script for SelectTopNRows command from SSMS  ******/


  CREATE VIEW cmn.game_statistics AS
  (
	  SELECT
		g.game_id, 
		g.start_date AS start_date,
		DATEDIFF(SECOND, g.start_date, last_move.end_time) AS seconds_played
	  FROM cmn.game g
	  CROSS APPLY (
		SELECT TOP 1 end_time
		FROM cmn.game_move gm
		WHERE gm.game_id = g.game_id 
		ORDER BY gm.game_move_id DESC
	  ) AS last_move
	  JOIN cmn.game_move gm ON gm.game_id = g.game_id
  )
