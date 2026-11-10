use cozy_chess::{Board, Color, File, Move, Piece, Rank, Square};
use std::ffi::{c_char, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

const TOKENS: usize = 64 * 12;
const MOVES: usize = 4352;

fn piece_number(piece: Piece) -> usize {
    match piece {
        Piece::Pawn => 1,
        Piece::Knight => 2,
        Piece::Bishop => 3,
        Piece::Rook => 4,
        Piece::Queen => 5,
        Piece::King => 6,
    }
}

fn promotion_number(piece: Piece) -> usize {
    match piece {
        Piece::Queen => 0,
        Piece::Rook => 1,
        Piece::Bishop => 2,
        Piece::Knight => 3,
        _ => unreachable!(),
    }
}

fn maia_move_index(board: &Board, mv: Move) -> usize {
    let black = board.side_to_move() == Color::Black;
    let from = mv.from;
    let moved = board.piece_on(from).expect("generated move has a piece");
    let mut to = mv.to;
    if moved == Piece::King && board.colors(board.side_to_move()).has(to) {
        let rank = Rank::First.relative_to(board.side_to_move());
        let file = if to.file() as u8 > from.file() as u8 {
            File::G
        } else {
            File::C
        };
        to = Square::new(file, rank);
    }
    let from_index = if black {
        from as usize ^ 56
    } else {
        from as usize
    };
    let to_index = if black { to as usize ^ 56 } else { to as usize };
    match mv.promotion {
        Some(piece) => {
            let from_file = from_index & 7;
            let to_file = to_index & 7;
            4096 + from_file * 32 + to_file * 4 + promotion_number(piece)
        }
        None => from_index * 64 + to_index,
    }
}

fn features(board: &Board, tokens: &mut [f32], mask: &mut [u8]) {
    tokens.fill(0.0);
    mask.fill(0);
    let black = board.side_to_move() == Color::Black;
    for color in [Color::White, Color::Black] {
        for piece in Piece::ALL {
            for square in board.colored_pieces(color, piece) {
                let square_index = if black {
                    square as usize ^ 56
                } else {
                    square as usize
                };
                let piece_color = if black { !color } else { color };
                let token = piece_number(piece) + if piece_color == Color::Black { 6 } else { 0 };
                tokens[square_index * 12 + token - 1] = 1.0;
            }
        }
    }
    board.generate_moves(|moves| {
        for mv in moves {
            mask[maia_move_index(board, mv)] = 1;
        }
        false
    });
}

/// # Safety
/// `fen` must be a NUL-terminated UTF-8 string. `tokens` and `mask` must point to writable
/// buffers of at least 768 f32 values and 4352 u8 values respectively.
#[no_mangle]
pub unsafe extern "C" fn levelrl_features(
    fen: *const c_char,
    tokens: *mut f32,
    mask: *mut u8,
) -> i32 {
    if fen.is_null() || tokens.is_null() || mask.is_null() {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let fen = match CStr::from_ptr(fen).to_str() {
            Ok(value) => value,
            Err(_) => return -2,
        };
        let board = match Board::from_fen(fen, false) {
            Ok(value) => value,
            Err(_) => return -3,
        };
        features(
            &board,
            std::slice::from_raw_parts_mut(tokens, TOKENS),
            std::slice::from_raw_parts_mut(mask, MOVES),
        );
        0
    }))
    .unwrap_or(-4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_position_has_twenty_moves_and_thirty_two_pieces() {
        let board = Board::startpos();
        let mut tokens = [0.0; TOKENS];
        let mut mask = [0; MOVES];
        features(&board, &mut tokens, &mut mask);
        assert_eq!(tokens.iter().filter(|&&x| x == 1.0).count(), 32);
        assert_eq!(mask.iter().filter(|&&x| x != 0).count(), 20);
        assert_eq!(
            mask["e2e4"
                .parse::<Move>()
                .map(|m| m.from as usize * 64 + m.to as usize)
                .unwrap()],
            1
        );
    }

    #[test]
    fn black_promotions_use_mirrored_maia_indices() {
        let board = Board::from_fen("k7/8/8/8/8/8/7p/4K3 b - - 0 1", false).unwrap();
        let mut tokens = [0.0; TOKENS];
        let mut mask = [0; MOVES];
        features(&board, &mut tokens, &mut mask);
        for (offset, piece) in [(0, "q"), (1, "r"), (2, "b"), (3, "n")] {
            assert_eq!(mask[4096 + 7 * 32 + 7 * 4 + offset], 1, "{piece}");
        }
    }
}
