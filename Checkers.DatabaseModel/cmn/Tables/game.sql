CREATE TABLE [cmn].[game] (
    [game_id]                     INT         IDENTITY (1, 1) NOT NULL,
    [white_player_information_id] INT         NOT NULL,
    [black_player_information_id] INT         NOT NULL,
    [game_type_id]                INT         NOT NULL,
    [game_size]                   INT         NOT NULL,
    [game_result]                 VARCHAR (1) NOT NULL,
    [move_count]                  INT         NOT NULL,
    [start_date]                  DATETIME    NOT NULL,
    PRIMARY KEY CLUSTERED ([game_id] ASC),
    CHECK ([game_result]='D' OR [game_result]='B' OR [game_result]='W'),
    CHECK ([game_result]='D' OR [game_result]='B' OR [game_result]='W'),
    CHECK ([game_result]='D' OR [game_result]='B' OR [game_result]='W'),
    FOREIGN KEY ([black_player_information_id]) REFERENCES [cmn].[player_information] ([player_information_id]),
    FOREIGN KEY ([black_player_information_id]) REFERENCES [cmn].[player_information] ([player_information_id]),
    FOREIGN KEY ([game_type_id]) REFERENCES [cmn].[game_type] ([game_type_id]),
    FOREIGN KEY ([game_type_id]) REFERENCES [cmn].[game_type] ([game_type_id]),
    FOREIGN KEY ([game_type_id]) REFERENCES [cmn].[game_type] ([game_type_id]),
    FOREIGN KEY ([white_player_information_id]) REFERENCES [cmn].[player_information] ([player_information_id]),
    FOREIGN KEY ([white_player_information_id]) REFERENCES [cmn].[player_information] ([player_information_id])
);


GO
CREATE NONCLUSTERED INDEX [IX_game_white_player_information_id_black_player_information_id_game_type_id_game_size]
    ON [cmn].[game]([white_player_information_id] ASC, [black_player_information_id] ASC, [game_type_id] ASC, [game_size] ASC);

