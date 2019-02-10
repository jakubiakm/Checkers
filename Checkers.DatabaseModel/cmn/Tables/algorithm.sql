CREATE TABLE [cmn].[algorithm] (
    [algorithm_id]   INT          IDENTITY (1, 1) NOT NULL,
    [algorithm_name] VARCHAR (50) NOT NULL,
    PRIMARY KEY CLUSTERED ([algorithm_id] ASC)
);


GO
CREATE UNIQUE NONCLUSTERED INDEX [UQ_algorithm_algorithm_name]
    ON [cmn].[algorithm]([algorithm_name] ASC);

