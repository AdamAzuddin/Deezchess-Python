import chess


# Function to calculate material balance
import chess


def material_balance(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,  # The king has no material value in this context
    }

    def is_center_open():
        # Central squares: e4, d4, e5, d5
        central_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        return all(
            board.piece_at(square) is None
            or board.piece_at(square).piece_type != chess.PAWN
            for square in central_squares
        )

    def calculate_material(color):
        material = 0
        bishop_count = len(list(board.pieces(chess.BISHOP, color)))
        has_bishop_pair = bishop_count == 2

        for piece_type in [
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
        ]:
            material += sum(
                piece_values.get(piece_type, 0)
                for square in board.pieces(piece_type, color)
            )

        # Adjust value for bishop pair if center is open
        if is_center_open() and has_bishop_pair:
            material += 0.5 * bishop_count  # Add 0.5 for each bishop in an open center

        return material

    white_material = calculate_material(chess.WHITE)
    black_material = calculate_material(chess.BLACK)

    return {"white_material": white_material, "black_material": black_material}


# Function to calculate the material imbalance between the two sides
def material_imbalance(board):
    white_material = material_balance(board)["white_material"]
    black_material = material_balance(board)["black_material"]
    
    return white_material - black_material


# Function to detect minor piece imbalances (e.g., bishop vs. knight)
def minor_piece_imbalance(board):
    white_knights = len(board.pieces(chess.KNIGHT, chess.WHITE))
    black_knights = len(board.pieces(chess.KNIGHT, chess.BLACK))
    white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))

    imbalance = abs(white_knights - black_knights) + abs(white_bishops - black_bishops)
    return 1 if imbalance > 0 else 0


# Function to evaluate the king's safety (castling and pawn protection)
def king_safety(board):
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)

    def count_pawns_around(king_square):
        pawn_protection = 0
        for square in [
            king_square - 1,
            king_square + 1,
            king_square - 8,
            king_square + 8,
        ]:  # adjacent squares
            if 0 <= square < 64 and board.piece_at(square) == chess.PAWN:
                pawn_protection += 1
        return pawn_protection

    white_pawn_protection = count_pawns_around(white_king_square)
    black_pawn_protection = count_pawns_around(black_king_square)

    return {
        "white_king_castled": 1 if board.has_castling_rights(chess.WHITE) else 0,
        "black_king_castled": 1 if board.has_castling_rights(chess.BLACK) else 0,
        "white_pawn_protection": white_pawn_protection,
        "black_pawn_protection": black_pawn_protection,
    }


# Function to evaluate various aspects of pawn structure
def pawn_structure(board):
    def is_isolated(square, color):
        file = square % 8
        adjacent_files = []
        if file > 0:
            adjacent_files.append(file - 1)
        if file < 7:
            adjacent_files.append(file + 1)
        return not any(
            any(
                board.piece_at(chess.square(adj_file, rank)) == chess.PAWN
                and board.color_at(chess.square(adj_file, rank)) == color
                for rank in range(8)
            )
            for adj_file in adjacent_files
        )

    def is_doubled(square, color):
        file = square % 8
        rank = square // 8
        return any(
            rank2 != rank
            and board.piece_at(chess.square(file, rank2)) == chess.PAWN
            and board.color_at(chess.square(file, rank2)) == color
            for rank2 in range(8)
        )

    def is_backward(square, color):
        file = square % 8
        rank = square // 8
        direction = -1 if color == chess.WHITE else 1
        return not any(
            board.piece_at(chess.square(adj_file, rank - direction)) == chess.PAWN
            and board.color_at(chess.square(adj_file, rank - direction)) == color
            for adj_file in [file - 1, file + 1]
            if 0 <= adj_file <= 7
        )

    def is_passed(square, color):
        file = square % 8
        rank = square // 8
        direction = -1 if color == chess.WHITE else 1
        opposing_color = not color
        return not any(
            board.piece_at(chess.square(adj_file, r)) == chess.PAWN
            and board.color_at(chess.square(adj_file, r)) == opposing_color
            for adj_file in [file - 1, file, file + 1]
            if 0 <= adj_file <= 7
            for r in range(
                rank + direction, 8 if color == chess.WHITE else -1, direction
            )
        )

    def is_connected(square, color):
        file = square % 8
        rank = square // 8
        return any(
            board.piece_at(chess.square(adj_file, adj_rank)) == chess.PAWN
            and board.color_at(chess.square(adj_file, adj_rank)) == color
            for adj_file in [file - 1, file + 1]
            if 0 <= adj_file <= 7
            for adj_rank in [rank - 1, rank, rank + 1]
            if 0 <= adj_rank <= 7
        )

    def count_pawn_structures(color):
        isolated_pawns = 0
        doubled_pawns = 0
        backward_pawns = 0
        passed_pawns = 0
        connected_pawns = 0

        for square in board.pieces(chess.PAWN, color):
            if is_isolated(square, color):
                isolated_pawns += 1
            if is_doubled(square, color):
                doubled_pawns += 1
            if is_backward(square, color):
                backward_pawns += 1
            if is_passed(square, color):
                passed_pawns += 1
            if is_connected(square, color):
                connected_pawns += 1

        return (
            isolated_pawns,
            doubled_pawns,
            backward_pawns,
            passed_pawns,
            connected_pawns,
        )

    # Calculate pawn structures for white and black
    white_isolated, white_doubled, white_backward, white_passed, white_connected = (
        count_pawn_structures(chess.WHITE)
    )
    black_isolated, black_doubled, black_backward, black_passed, black_connected = (
        count_pawn_structures(chess.BLACK)
    )

    # Return all pawn structures as individual values
    return {
        "white_isolated_pawns": white_isolated,
        "white_doubled_pawns": white_doubled,
        "white_backward_pawns": white_backward,
        "white_passed_pawns": white_passed,
        "white_connected_pawns": white_connected,
        "black_isolated_pawns": black_isolated,
        "black_doubled_pawns": black_doubled,
        "black_backward_pawns": black_backward,
        "black_passed_pawns": black_passed,
        "black_connected_pawns": black_connected,
    }


# Function to calculate control over central squares
def center_control(board, is_white_player):
    central_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    player_control = 0

    for square in central_squares:
        if is_white_player:
            # Add 1 for each square attacked by white
            if board.is_attacked_by(chess.WHITE, square):
                player_control += 1
        else:
            # Add 1 for each square attacked by black
            if board.is_attacked_by(chess.BLACK, square):
                player_control += 1

    return player_control


# Function to calculate the number of open files for rooks
def open_files(board):
    open_files = 0
    for file in range(8):
        if all(board.piece_at(file + 8 * rank) is None for rank in range(1, 7)):
            open_files += 1
    return open_files


# Function to calculate the number of semi-open files for rooks
def semi_open_files(board):
    white_semi_open_files = 0
    black_semi_open_files = 0

    for file in range(8):  # Iterate over all 8 files (0 to 7)
        has_white_pawn = False
        has_black_pawn = False

        for rank in range(8):  # Check all ranks in the current file
            square = file + 8 * rank
            piece = board.piece_at(square)
            if piece:
                if piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        has_white_pawn = True
                    else:
                        has_black_pawn = True

        # If there are no black pawns on the file, it's semi-open for white
        if not has_black_pawn and has_white_pawn:
            white_semi_open_files += 1

        # If there are no white pawns on the file, it's semi-open for black
        if not has_white_pawn and has_black_pawn:
            black_semi_open_files += 1

    return {
        "white_semi_open_files": white_semi_open_files,
        "black_semi_open_files": black_semi_open_files,
    }


# Function to calculate the total number of legal moves for all pieces
def piece_mobility(board):
    white_mobility = 0
    black_mobility = 0

    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type != chess.PAWN:
            if piece.color == chess.WHITE:
                white_mobility += 1
            else:
                black_mobility += 1

    return {
        "white_piece_mobility": white_mobility,
        "black_piece_mobility": black_mobility,
    }


# Function to calculate the number of pieces actively attacking opponent squares
def piece_activity(board):
    white_activity = 0
    black_activity = 0

    # Iterate through pieces on the board
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN:  # Exclude pawns
            continue

        if piece.color == chess.WHITE:
            # Check if the piece is attacking any black pieces
            if any(
                board.is_attacked_by(chess.BLACK, move.to_square)
                for move in board.legal_moves
                if move.from_square == square
            ):
                white_activity += 1
        else:
            # Check if the piece is attacking any white pieces
            if any(
                board.is_attacked_by(chess.WHITE, move.to_square)
                for move in board.legal_moves
                if move.from_square == square
            ):
                black_activity += 1

    return {
        "white_piece_activity": white_activity,
        "black_piece_activity": black_activity,
    }


# Function to calculate king activity in the endgame (distance to the center or passed pawns)
def king_activity_endgame(board):
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)

    white_king_dist = abs(white_king_square // 8 - 3) + abs(
        white_king_square % 8 - 3
    )  # to e4 (center)
    black_king_dist = abs(black_king_square // 8 - 3) + abs(
        black_king_square % 8 - 3
    )  # to e5 (center)

    return {
        "white_king_dist_to_center": white_king_dist,
        "black_king_dist_to_center": black_king_dist,
    }


# Function to calculate the number of pieces supporting each other
def piece_coordination(board):
    white_coordination = 0
    black_coordination = 0

    # Iterate over pieces on the board
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN:  # Exclude pawns
            continue

        for move in board.legal_moves:
            if move.from_square == square:
                if (
                    move.to_square != square
                    and board.piece_at(move.to_square) is not None
                ):
                    # Check if the target square is occupied by a friendly piece
                    if board.piece_at(move.to_square).color == piece.color:
                        if piece.color == chess.WHITE:
                            white_coordination += 1
                        else:
                            black_coordination += 1

    return {
        "white_coordination": white_coordination,
        "black_coordination": black_coordination,
    }


# Function to calculate the number of pieces attacking opponent pieces
def threats(board):
    white_attacking = 0
    black_attacking = 0
    white_hanging = 0
    black_hanging = 0

    # Iterate over pieces on the board
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN:  # Exclude pawns
            continue

        if piece.color == chess.WHITE:
            # Count attacking pieces
            if board.is_attacked_by(chess.BLACK, square):
                white_attacking += 1
            # Count hanging pieces
            if not board.is_attacked_by(chess.WHITE, square):
                white_hanging += 1

        elif piece.color == chess.BLACK:
            # Count attacking pieces
            if board.is_attacked_by(chess.WHITE, square):
                black_attacking += 1
            # Count hanging pieces
            if not board.is_attacked_by(chess.BLACK, square):
                black_hanging += 1

    return {
        "white_attacking_pieces": white_attacking,
        "black_attacking_pieces": black_attacking,
        "white_hanging_pieces": white_hanging,
        "black_hanging_pieces": black_hanging,
    }


# Function to calculate space advantage (total squares controlled in opponent's territory)
def space_advantage(board, is_white):
    controlled_squares = 0

    # Define the range of squares that represent the opponent's territory
    opponent_territory_start = 32  # For black: 32 to 63; for white: 0 to 31
    opponent_territory_end = 64 if is_white else 32

    # Iterate over the squares in the opponent's territory
    for square in range(opponent_territory_start, opponent_territory_end):
        if is_white:  # Calculate for white's advantage
            if board.is_attacked_by(chess.WHITE, square):
                controlled_squares += 1
            if board.is_attacked_by(chess.BLACK, square):
                controlled_squares -= 1
        else:  # Calculate for black's advantage
            if board.is_attacked_by(chess.BLACK, square):
                controlled_squares += 1
            if board.is_attacked_by(chess.WHITE, square):
                controlled_squares -= 1

    return controlled_squares


# Function to check if one side has a bishop pair
def bishop_pair(board):
    white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    return {
        "white_bishop_pair": 1 if white_bishops == 2 else 0,
        "black_bishop_pair": 1 if black_bishops == 2 else 0,
    }


# Function to calculate the number of knights on strong outpost squares
def knight_outposts(board, is_white):
    knight_outposts = 0
    outpost_squares = [
        chess.D3,
        chess.E3,
        chess.D6,
        chess.E6,
    ]  # Example strong outpost squares

    # Iterate through the squares on the board
    for square in range(64):
        piece = board.piece_at(square)

        if piece and piece.piece_type == chess.KNIGHT:  # Check if the piece is a knight
            if (
                piece.color == chess.WHITE and is_white
            ):  # If the knight is white and we are calculating for white
                if square in outpost_squares:
                    knight_outposts += 1
            elif (
                piece.color == chess.BLACK and not is_white
            ):  # If the knight is black and we are calculating for black
                if square in outpost_squares:
                    knight_outposts += 1

    return knight_outposts


# Function to calculate the number of rooks on the opponent's seventh rank
def rook_on_seventh_rank(board):
    seventh_rank_rooks = 0
    for rook_square in board.pieces(chess.ROOK, chess.WHITE):
        if rook_square in [x for x in range(56, 64)]:
            seventh_rank_rooks += 1
    return seventh_rank_rooks


# Function to calculate pawn majority
def pawn_majority(board):
    white_pawn_majority = 0
    black_pawn_majority = 0
    for square in board.pieces(chess.PAWN, chess.WHITE):
        if square % 8 > 3:
            white_pawn_majority += 1
    for square in board.pieces(chess.PAWN, chess.BLACK):
        if square % 8 < 3:
            black_pawn_majority += 1
    return {
        "white_pawn_majority": white_pawn_majority,
        "black_pawn_majority": black_pawn_majority,
    }


# Function to calculate passed pawn advancement
def is_passed_pawn(board, square, color):
    """Check if a pawn on the given square is a passed pawn."""
    pawn_file = chess.square_file(square)
    opponent_color = not color

    # Get squares in the same file and adjacent files ahead of the pawn
    direction = 1 if color == chess.WHITE else -1
    forward_squares = [
        chess.square(file, rank)
        for file in range(max(0, pawn_file - 1), min(7, pawn_file + 1) + 1)
        for rank in range(
            chess.square_rank(square) + direction,
            (8 if color == chess.WHITE else -1),
            direction,
        )
    ]

    # Check if there are any opponent pawns in these squares
    for target_square in forward_squares:
        if board.piece_at(target_square) == chess.Piece(chess.PAWN, opponent_color):
            return False
    return True


def passed_pawn_advancement(board, is_white_player):
    advancement = 0

    if is_white_player:
        # Check each pawn for White
        for square in board.pieces(chess.PAWN, chess.WHITE):
            if is_passed_pawn(board, square, chess.WHITE):
                advancement += chess.square_rank(
                    square
                )  # Reward advancement rank (0-7)
    else:
        # Check each pawn for Black
        for square in board.pieces(chess.PAWN, chess.BLACK):
            if is_passed_pawn(board, square, chess.BLACK):
                advancement += 7 - chess.square_rank(
                    square
                )  # Reward advancement rank (7-0)

    return advancement


"""
Material Balance: Total material value for each side.
Material Imbalance: Difference in total material value.
Minor Piece Imbalance: Presence of minor piece imbalances (e.g., bishop vs. knight).
King Safety: Is the king castled (yes/no)?
King Safety: Number of pawns protecting the king.
Pawn Structure: Number of isolated pawns.
Pawn Structure: Number of doubled pawns.
Pawn Structure: Number of backward pawns.
Pawn Structure: Presence of passed pawns.
Pawn Structure: Number of connected pawns.
Center Control: Number of pawns and pieces controlling central squares (e4, d4, e5, d5).
Open Files: Number of open files for rooks.
Semi-Open Files: Number of semi-open files for rooks.
Piece Mobility: Total number of legal moves available for all pieces.
Piece Activity: Number of pieces actively attacking opponent squares.
King Activity (Endgame): King distance to the center (Manhattan distance).
King Activity (Endgame): King distance to passed pawns (Manhattan distance).

Piece Coordination: Number of pieces supporting each other.
Threats: Number of pieces attacking opponent pieces.
Threats: Number of unprotected pieces (hanging pieces).
Space Advantage: Total squares controlled in opponent's territory.
Bishop Pair: Does one side have a bishop pair (yes/no)?
Knight Outposts: Number of knights on strong outpost squares.
Rook Activity: Number of rooks on open or semi-open files.
Rook on Seventh Rank: Number of rooks on the opponent's 7th rank.
Pawn Majority: Presence of a pawn majority on one side.
Passed Pawn Advancement: Distance of passed pawns from promotion (in ranks).
King Zone Weakness: Number of weak squares around the king.
Queen Activity: Number of legal moves available for the queen.
"""
