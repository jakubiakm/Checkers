using Checkers.Logic.Enums;
using Checkers.Logic.GameObjects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.Engines
{
    public interface IEngine
    {
        EngineKind Kind { get; }

        Move MakeMove(CheckersBoard currentBoard);


    }
}
