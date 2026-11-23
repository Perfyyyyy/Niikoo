#pragma once
// Piece values (the eval is net only, the old piece-square tables are gone)

inline constexpr int PIECE_VALUE[6] = {100, 320, 330, 500, 900, 0};
inline constexpr int ORDER_VALUE[6] = {1, 3, 3, 5, 9, 0};  // for move ordering