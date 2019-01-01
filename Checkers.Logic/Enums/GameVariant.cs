using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Checkers.Logic.Enums
{
    public enum GameVariant
    {
        [Description("Warcaby")]
        Checkers,

        [Description("Antywarcaby")]
        Anticheckers
    }
}
