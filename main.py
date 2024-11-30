import chess.pgn
import csv
import os
import argparse
from tqdm import tqdm

from extract_features import (
    material_balance,
    material_imbalance,
    minor_piece_imbalance,
    king_safety,
    pawn_structure,
    center_control,
    open_files,
    semi_open_files,
    piece_mobility,
    piece_activity,
    king_activity_endgame,
    piece_coordination,
    threats,
    space_advantage,
    bishop_pair,
    knight_outposts,
    rook_on_seventh_rank,
    pawn_majority,
    passed_pawn_advancement,
)


# Extract the player's name from the PGN file name
def extract_player_name_from_filename(pgn_filename):
    return os.path.splitext(os.path.basename(pgn_filename))[0]


# Function to identify if the current player is white
def is_player_white(game, player_name):
    return game.headers["White"] == player_name

parser = argparse.ArgumentParser(description="Convert PGN file to csv file of features for each possible position of every game in the file")
parser.add_argument("input_file", type=str, help="The path to the input PGN file")
parser.add_argument("output_file", type=str, help="The path to the output csv file")
args = parser.parse_args()


pgn_file = args.input_file
output_file = args.output_file
player_name = extract_player_name_from_filename(pgn_file)

num_of_games = 0

## Task!
# -1. Only extract the first 10 moves
# -2. Convert the FEN to array of 64 int, 1=white pawn, -1= black pawn, 0=empty square, white knight = 2, etc
# -3. Label the move played in the game as 1, other legal moves as 0
# -4. Experiment with models: Random forest, Xgboost, Balanced Random Forest, neural networks, SVM
# -5. Split into train,test and val set
# -6. Train the models
# -7. Evaluate key metrics like accuracy, precision, recall and F1 score.
# 10. Extract another csv files for middlegame and endgame with features like pawn structure, king safety, piece activity, open or close position, number of open files
## For definition
# Accuracy: How often the model is correct (good for balanced data).
# Precision: How accurate the model is when it predicts positive cases.
# Recall: How many actual positive cases the model successfully identifies.

data = []
num_of_positions = 0
print(f"Opening files...")
with open(pgn_file) as pgn, open(output_file, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header with columns for each feature
    csv_writer.writerow(
        [
            "is_white_player",
            "position_fen",
            "move",
            "white_material_balance",
            "black_material_balance",
            "material_imbalance",
            "minor_piece_imbalance",
            "white_king_castled",
            "black_king_castled",
            "white_pawn_protection",
            "black_pawn_protection",
            "white_isolated_pawns",
            "white_doubled_pawns",
            "white_backward_pawns",
            "white_passed_pawns",
            "white_connected_pawns",
            "black_isolated_pawns",
            "black_doubled_pawns",
            "black_backward_pawns",
            "black_passed_pawns",
            "black_connected_pawns",
            "center_control",
            "open_files",
            "white_semi_open_files",
            "black_semi_open_files",
            "white_piece_mobility",
            "black_piece_mobility",
            "white_piece_activity",
            "black_piece_activity",
            "king_activity_endgame_center",
            "king_activity_endgame_passed_pawns",
            "white_coordination",
            "black_coordination",
            "white_attacking_pieces",
            "white_hanging_pieces",
            "black_attacking_pieces",
            "black_hanging_pieces",
            "player_space_advantage",
            "white_bishop_pair",
            "black_bishop_pair",
            "player_knight_outposts",
            "rook_on_seventh_rank",
            "white_pawn_majority",
            "black_pawn_majority",
            "player_passed_pawn_advancement",
            "label",
        ]
    )
    pbar = tqdm(desc="Extracting features", unit=" games")
    while True:
        game = chess.pgn.read_game(pgn)

        pbar.update(1)

        if game is None:
            break

        num_of_games += 1
        num_of_moves = 0
        is_white = is_player_white(game, player_name)

        board = game.board()
        for move in game.mainline_moves():
            num_of_moves += 1

            # Skip the first 10 moves (opening phase)
            if num_of_moves <= 10:
                board.push(move)
                continue

            fen = board.fen()
            legal_moves = list(board.legal_moves)
            material_balance_features = material_balance(board)
            king_safety_features = king_safety(board)
            pawn_structure_features = pawn_structure(board)
            semi_open_files_features = semi_open_files(board)
            piece_mobility_features = piece_mobility(board)
            piece_activity_features = piece_activity(board)
            king_activity_endgame_features = king_activity_endgame(board)
            piece_coordination_features = piece_coordination(board)
            threats_features = threats(board)
            bishop_pair_features = bishop_pair(board)
            pawn_majority_features = pawn_majority(board)

            for legal_move in legal_moves:
                simulated_board = board.copy()
                simulated_board.push(legal_move)
                # Get feature values for the current board state
                features = [
                    1 if is_white else 0,
                    fen,
                    legal_move.uci(),  # move in UCI notation
                    material_balance_features["white_material"],
                    material_balance_features["black_material"],
                    material_imbalance(board),  # Material imbalance
                    minor_piece_imbalance(board),  # Minor piece imbalance
                    king_safety_features[
                        "white_king_castled"
                    ],  # White king castled status
                    king_safety_features[
                        "black_king_castled"
                    ],  # Black king castled status
                    king_safety_features[
                        "white_pawn_protection"
                    ],  # White pawn protection count
                    king_safety_features[
                        "black_pawn_protection"
                    ],  # Black pawn protection count
                    pawn_structure_features["white_isolated_pawns"],
                    pawn_structure_features["white_doubled_pawns"],
                    pawn_structure_features["white_backward_pawns"],
                    pawn_structure_features["white_passed_pawns"],
                    pawn_structure_features["white_connected_pawns"],
                    pawn_structure_features["black_isolated_pawns"],
                    pawn_structure_features["black_doubled_pawns"],
                    pawn_structure_features["black_backward_pawns"],
                    pawn_structure_features["black_passed_pawns"],
                    pawn_structure_features["black_connected_pawns"],
                    center_control(board, is_white),  # Center control
                    open_files(board),  # Open files
                    semi_open_files_features["white_semi_open_files"],
                    semi_open_files_features["black_semi_open_files"],
                    piece_mobility_features["white_piece_mobility"],
                    piece_mobility_features["black_piece_mobility"],
                    piece_activity_features["white_piece_activity"],
                    piece_activity_features["black_piece_activity"],
                    king_activity_endgame_features["white_king_dist_to_center"],
                    king_activity_endgame_features["black_king_dist_to_center"],
                    piece_coordination_features["white_coordination"],
                    piece_coordination_features["black_coordination"],
                    threats_features["white_attacking_pieces"],
                    threats_features["white_hanging_pieces"],
                    threats_features["black_attacking_pieces"],
                    threats_features["black_hanging_pieces"],
                    space_advantage(board, is_white),  # Space advantage
                    bishop_pair_features["white_bishop_pair"],
                    bishop_pair_features["black_bishop_pair"],
                    knight_outposts(board, is_white),  # Number of knight outposts
                    rook_on_seventh_rank(board),  # Number of rooks on seventh rank
                    pawn_majority_features[
                        "white_pawn_majority"
                    ],  # White pawn majority
                    pawn_majority_features[
                        "black_pawn_majority"
                    ],  # Black pawn majority
                    passed_pawn_advancement(board, is_white),  # Passed pawn advancement
                    (
                        1 if legal_move == move else 0
                    ),  # label (1 if the move is actually made, 0 otherwise)
                ]

                # Write the data to the CSV
                csv_writer.writerow(features)
                num_of_positions += 1
                simulated_board.pop()

            board.push(move)
pbar.close()
print("Finished extracting features")
print(f"Number of positions: {num_of_positions}")
print(f"Number of games: {num_of_games}")
print(f"Extracted features from {pgn_file} to {output_file}")
