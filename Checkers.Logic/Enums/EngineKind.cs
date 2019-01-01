using System;
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

        [Description("NVIDIA CUDA MCTS UCT")]
        Cuda
    }
}
