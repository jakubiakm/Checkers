﻿using Checkers.Logic.Enums;
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
        /// <summary>
        /// Rodzaj silnika
        /// </summary>
        EngineKind Kind { get; }

        /// <summary>
        /// Kolor gracza
        /// </summary>
        PieceColor Color { get; set; }

        /// <summary>
        /// Wygeneruj następny ruch
        /// </summary>
        /// <param name="currentBoard">Aktualna plansza na której ma być wygenerowany ruch</param>
        /// <param name="variant">Wariant gry dla którego ma być znaleziony ruch</param>
        /// <param name="gameMoves">Lista ostatnio wykonanych ruchów w grze (służy do sprawdzenia czy był remis)</param>
        /// <returns></returns>
        Move MakeMove(CheckersBoard currentBoard, GameVariant variant, List<Move> gameMoves);

        /// <summary>
        /// Resetowanie silnika - po każdej rozegranej grze wartości silnika powinny zostać zresetowane (jak np. ziarno do generatora liczb pseudolosowych)
        /// </summary>
        void Reset();
    }
}
