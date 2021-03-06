﻿CREATE TABLE [cmn].[player_information] (
    [player_information_id] INT        IDENTITY (1, 1) NOT NULL,
    [player_id]             INT        NOT NULL,
    [algorithm_id]          INT        NOT NULL,
    [number_of_pieces]      INT        NOT NULL,
    [tree_depth]            INT        NULL,
    [uct_parameter]         FLOAT (53) NULL,
    [number_of_iterations]  INT        NULL,
    PRIMARY KEY CLUSTERED ([player_information_id] ASC),
    FOREIGN KEY ([algorithm_id]) REFERENCES [cmn].[algorithm] ([algorithm_id]),
    FOREIGN KEY ([algorithm_id]) REFERENCES [cmn].[algorithm] ([algorithm_id]),
    FOREIGN KEY ([algorithm_id]) REFERENCES [cmn].[algorithm] ([algorithm_id]),
    FOREIGN KEY ([algorithm_id]) REFERENCES [cmn].[algorithm] ([algorithm_id]),
    FOREIGN KEY ([algorithm_id]) REFERENCES [cmn].[algorithm] ([algorithm_id]),
    FOREIGN KEY ([player_id]) REFERENCES [cmn].[player] ([player_id]),
    FOREIGN KEY ([player_id]) REFERENCES [cmn].[player] ([player_id]),
    FOREIGN KEY ([player_id]) REFERENCES [cmn].[player] ([player_id]),
    FOREIGN KEY ([player_id]) REFERENCES [cmn].[player] ([player_id]),
    FOREIGN KEY ([player_id]) REFERENCES [cmn].[player] ([player_id]),
    FOREIGN KEY ([player_id]) REFERENCES [cmn].[player] ([player_id]),
    FOREIGN KEY ([player_id]) REFERENCES [cmn].[player] ([player_id]),
    FOREIGN KEY ([player_id]) REFERENCES [cmn].[player] ([player_id])
);





