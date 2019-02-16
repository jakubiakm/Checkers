using Checkers.Logic.Engines;
using Checkers.Logic.Enums;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Data;
using System.Windows.Markup;

namespace Checkers.UI.ViewModel
{
    public class SettingsViewModel : INotifyPropertyChanged
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

            #region CUDA ENGINE
            WhitePlayerCudaEngineBlockSize = 10;
            WhitePlayerCudaEngineGridSize = 10;
            WhitePlayerCudaEngineMctsIteration = 25;
            BlackPlayerCudaEngineBlockSize = 75;
            BlackPlayerCudaEngineGridSize = 75;
            BlackPlayerCudaEngineMctsIteration = 25;
            #endregion

            WhitePlayerMinmaxEngineTreeDepth = 5;
            BlackPlayerMinmaxEngineTreeDepth = 5;

            switch (whitePlayerEngine.Kind)
            {
                case EngineKind.Random:
                    var randomEngine = (RandomEngine)whitePlayerEngine;
                    WhitePlayerRandomEngineUseRandomSeed = randomEngine.Seed == null;
                    WhitePlayerRandomEngineSeedValue = randomEngine.Seed ?? 0;
                    break;
                case EngineKind.Cuda:
                    var cudaEngine = (CudaEngine)whitePlayerEngine;
                    WhitePlayerCudaEngineBlockSize = cudaEngine.BlockSize;
                    WhitePlayerCudaEngineGridSize = cudaEngine.GridSize;
                    WhitePlayerCudaEngineMctsIteration = cudaEngine.MctsIterationCount;
                    break;
                case EngineKind.MinMax:
                    var minmaxEngine = (MinMaxEngine)whitePlayerEngine;
                    WhitePlayerMinmaxEngineTreeDepth = minmaxEngine.MinmaxTreeDepth;
                    break;
            }

            switch (blackPlayerEngine.Kind)
            {
                case EngineKind.Random:
                    var randomEngine = (RandomEngine)blackPlayerEngine;
                    BlackPlayerRandomEngineUseRandomSeed = randomEngine.Seed == null;
                    BlackPlayerRandomEngineSeedValue = randomEngine.Seed ?? 0;
                    break;
                case EngineKind.Cuda:
                    var cudaEngine = (CudaEngine)blackPlayerEngine;
                    BlackPlayerCudaEngineBlockSize = cudaEngine.BlockSize;
                    BlackPlayerCudaEngineGridSize = cudaEngine.GridSize;
                    BlackPlayerCudaEngineMctsIteration = cudaEngine.MctsIterationCount;
                    break;
                case EngineKind.MinMax:
                    var minmaxEngine = (MinMaxEngine)blackPlayerEngine;
                    BlackPlayerMinmaxEngineTreeDepth = minmaxEngine.MinmaxTreeDepth;
                    break;
            }
        }

        public int BoardSize
        {
            get
            {
                return boardSize;
            }
            set
            {
                boardSize = value;
                NotifyPropertyChanged();
            }
        }

        public int WhitePiecesCount
        {
            get
            {
                return whitePiecesCount;
            }
            set
            {
                whitePiecesCount = value;
                NotifyPropertyChanged();
            }
        }

        public int BlackPiecesCount
        {
            get
            {
                return blackPiecesCount;
            }
            set
            {
                blackPiecesCount = value;
                NotifyPropertyChanged();
            }
        }

        public int MoveAnimationTime { get; set; }

        public EngineKind WhitePlayerEngineKind { get; set; }

        public EngineKind BlackPlayerEngineKind { get; set; }

        public GameVariant CurrentGameVariant { get; set; }

        public bool WhitePlayerRandomEngineUseRandomSeed { get; set; }

        public bool BlackPlayerRandomEngineUseRandomSeed { get; set; }

        public int WhitePlayerRandomEngineSeedValue { get; set; }

        public int BlackPlayerRandomEngineSeedValue { get; set; }

        public int WhitePlayerMinmaxEngineTreeDepth
        {
            get
            {
                return whitePlayerMinmaxEngineTreeDepth;
            }
            set
            {
                whitePlayerMinmaxEngineTreeDepth = value;
                NotifyPropertyChanged();
            }
        }

        public int BlackPlayerMinmaxEngineTreeDepth
        {
            get
            {
                return blackPlayerMinmaxEngineTreeDepth;
            }
            set
            {
                blackPlayerMinmaxEngineTreeDepth = value;
                NotifyPropertyChanged();
            }
        }

        public int WhitePlayerCudaEngineMctsIteration
        {
            get
            {
                return whitePlayerCudaEngineMctsIteration;
            }
            set
            {
                whitePlayerCudaEngineMctsIteration = value;
                NotifyPropertyChanged();
            }
        }

        public int WhitePlayerCudaEngineGridSize
        {
            get
            {
                return whitePlayerCudaEngineGridSize;
            }
            set
            {
                whitePlayerCudaEngineGridSize = value;
                NotifyPropertyChanged();
            }
        }

        public int WhitePlayerCudaEngineBlockSize
        {
            get
            {
                return whitePlayerCudaEngineBlockSize;
            }
            set
            {
                whitePlayerCudaEngineBlockSize = value;
                NotifyPropertyChanged();
            }
        }

        public int BlackPlayerCudaEngineMctsIteration
        {
            get
            {
                return blackPlayerCudaEngineMctsIteration;
            }
            set
            {
                blackPlayerCudaEngineMctsIteration = value;
                NotifyPropertyChanged();
            }
        }

        public int BlackPlayerCudaEngineGridSize
        {
            get
            {
                return blackPlayerCudaEngineGridSize;
            }
            set
            {
                blackPlayerCudaEngineGridSize = value;
                NotifyPropertyChanged();
            }
        }

        public int BlackPlayerCudaEngineBlockSize
        {
            get
            {
                return blackPlayerCudaEngineBlockSize;
            }
            set
            {
                blackPlayerCudaEngineBlockSize = value;
                NotifyPropertyChanged();
            }
        }

        private int boardSize;

        private int whitePiecesCount;

        private int blackPiecesCount;

        private int whitePlayerMinmaxEngineTreeDepth;

        private int blackPlayerMinmaxEngineTreeDepth;

        private int whitePlayerCudaEngineMctsIteration;

        private int whitePlayerCudaEngineGridSize;

        private int whitePlayerCudaEngineBlockSize;

        private int blackPlayerCudaEngineMctsIteration;

        private int blackPlayerCudaEngineGridSize;

        private int blackPlayerCudaEngineBlockSize;

        public event PropertyChangedEventHandler PropertyChanged;

        // This method is called by the Set accessor of each property.  
        // The CallerMemberName attribute that is applied to the optional propertyName  
        // parameter causes the property name of the caller to be substituted as an argument.  
        private void NotifyPropertyChanged([CallerMemberName] String propertyName = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
