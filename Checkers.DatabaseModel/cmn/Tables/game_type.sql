CREATE TABLE [cmn].[game_type] (
    [game_type_id]   INT          IDENTITY (1, 1) NOT NULL,
    [game_type_name] VARCHAR (50) NULL,
    PRIMARY KEY CLUSTERED ([game_type_id] ASC)
);


GO
CREATE UNIQUE NONCLUSTERED INDEX [UQ_game_type_game_type_name]
    ON [cmn].[game_type]([game_type_name] ASC);

