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
        string GetName();

        Move MakeMove(CheckersBoard currentBoard);
    }
}
