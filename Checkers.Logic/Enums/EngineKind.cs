﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.Enums
{
    public enum EngineKind
    {
        [Description("Człowiek")]
        Human,

        [Description("Losowy")]
        Random,

        [Description("Nvidia cuda mcts")]
        Cuda,

        [Description("AlphaBeta")]
        AlphaBeta,

        [Description("MCTS z metodą UCT")]
        Mcts
    }
}
