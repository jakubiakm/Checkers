CREATE TABLE [cmn].[game_move] (
    [game_move_id]        INT           IDENTITY (1, 1) NOT NULL,
    [game_id]             INT           NOT NULL,
    [player]              VARCHAR (1)   NOT NULL,
    [start_time]          DATETIME      NOT NULL,
    [end_time]            DATETIME      NOT NULL,
    [from_position]       INT           NOT NULL,
    [to_position]         INT           NOT NULL,
    [beated_pieces_count] INT           NOT NULL,
    [beated_pieces]       VARCHAR (MAX) NULL,
    [board_after_move]    VARCHAR (MAX) NULL,
    PRIMARY KEY CLUSTERED ([game_move_id] ASC),
    CHECK ([player]='B' OR [player]='W'),
    CHECK ([player]='B' OR [player]='W'),
    CHECK ([player]='B' OR [player]='W'),
    FOREIGN KEY ([game_id]) REFERENCES [cmn].[game] ([game_id]),
    FOREIGN KEY ([game_id]) REFERENCES [cmn].[game] ([game_id])
);




GO
CREATE NONCLUSTERED INDEX [IX_game_move_game_id_player_start_time]
    ON [cmn].[game_move]([game_id] ASC, [player] ASC, [start_time] ASC, [end_time] ASC);

