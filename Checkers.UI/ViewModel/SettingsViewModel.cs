using Checkers.Logic.Engines;
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
            IEngine whitePlayerEngine,
            IEngine blackPlayerEngine,
            GameVariant currentGameVariant,
            int moveAnimationTime)
        {
            BoardSize = boardSize;
            WhitePiecesCount = whitePiecesCount;
            BlackPiecesCount = blackPiecesCount;
            WhitePlayerEngineKind = whitePlayerEngine.Kind;
            BlackPlayerEngineKind = blackPlayerEngine.Kind;
            CurrentGameVariant = currentGameVariant;
            MoveAnimationTime = moveAnimationTime;
            WhitePlayerRandomEngineUseRandomSeed = true;
            BlackPlayerRandomEngineUseRandomSeed = true;
            switch (whitePlayerEngine.Kind)
            {
                case EngineKind.Random:
                    var engine = (RandomEngine)whitePlayerEngine;
                    WhitePlayerRandomEngineUseRandomSeed = engine.Seed == null;
                    WhitePlayerRandomEngineSeedValue = engine.Seed ?? 0;
                    break;
            }

            switch (blackPlayerEngine.Kind)
            {
                case EngineKind.Random:
                    var engine = (RandomEngine)blackPlayerEngine;
                    BlackPlayerRandomEngineUseRandomSeed = engine.Seed == null;
                    BlackPlayerRandomEngineSeedValue = engine.Seed ?? 0;
                    break;
            }
        }

        public int BoardSize { get; set; }

        public int WhitePiecesCount { get; set; }

        public int BlackPiecesCount { get; set; }

        public int MoveAnimationTime { get; set; }

        public EngineKind WhitePlayerEngineKind { get; set; }

        public EngineKind BlackPlayerEngineKind { get; set; }

        public GameVariant CurrentGameVariant { get; set; }

        public bool WhitePlayerRandomEngineUseRandomSeed { get; set; }

        public bool BlackPlayerRandomEngineUseRandomSeed { get; set; }

        public int WhitePlayerRandomEngineSeedValue { get; set; }

        public int BlackPlayerRandomEngineSeedValue { get; set; }
    }
}
