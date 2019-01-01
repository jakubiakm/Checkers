using Checkers.Logic.Enums;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Data;
using System.Windows.Markup;

namespace Checkers.UI.ViewModel
{


    public class SettingsViewModel
    {
        public SettingsViewModel(
            int boardSize,
            int whitePiecesCount,
            int blackPiecesCount,
            EngineKind whitePlayerEngine,
            EngineKind blackPlayerEngine,
            GameVariant currentGameVariant,
            int moveAnimationTime)
        {
            BoardSize = boardSize;
            WhitePiecesCount = whitePiecesCount;
            BlackPiecesCount = blackPiecesCount;
            WhitePlayerEngineKind = whitePlayerEngine;
            BlackPlayerEngineKind = blackPlayerEngine;
            CurrentGameVariant = currentGameVariant;
            MoveAnimationTime = moveAnimationTime;
        }

        public int BoardSize { get; set; }

        public int WhitePiecesCount { get; set; }

        public int BlackPiecesCount { get; set; }

        public int MoveAnimationTime { get; set; }

        public EngineKind WhitePlayerEngineKind { get; set; }

        public EngineKind BlackPlayerEngineKind { get; set; }

        public GameVariant CurrentGameVariant { get; set; }
    }
}
