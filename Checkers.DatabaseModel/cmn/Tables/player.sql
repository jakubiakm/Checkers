CREATE TABLE [cmn].[player] (
    [player_id]   INT          IDENTITY (1, 1) NOT NULL,
    [player_name] VARCHAR (50) NOT NULL,
    PRIMARY KEY CLUSTERED ([player_id] ASC)
);


GO
CREATE UNIQUE NONCLUSTERED INDEX [UQ_player_player_name]
    ON [cmn].[player]([player_name] ASC);

