#!/usr/bin/python3

import os
import pickle
import numpy as np
from mlp import MLP


WHITE_WIN = 0
BLACK_WIN = 1
# PAT = 1
DATA_PATH = 'datasets'

def load(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def parse_chess_data(filename, hot_one=False):
    with open(filename, 'r') as file:
        lines = file.readlines()

    results = []
    checkmates = []
    fens = []

    for line in lines:
        if line.startswith('RES:'):
            result = line.split()[1]
            if result == '1-0':
                results.append(WHITE_WIN)
                skip = False
            elif result == '0-1':
                results.append(BLACK_WIN)
                skip = False
            else:
                skip = True
                continue
                # results.append(PAT)

        elif line.startswith('CHECKMATE:'):
            checkmate = line.split()[1] == 'True'
            checkmates.append(checkmate)

        elif line.startswith('FEN:'):
            fen = line.split(' ', 1)[1].strip()
            if hot_one == False:
                data = evaluer_echiquier(fen)
            else:
                data = fen_to_one_hot(fen)
            if skip == False:
                fens.append(data)

    return results, checkmates, fens

def fen_to_one_hot(fen):
    piece_to_index = {'r': 0, 'n': 1, 'b': 2, 'q': 3, 'k': 4, 'p': 5,
                      'R': 6, 'N': 7, 'B': 8, 'Q': 9, 'K': 10, 'P': 11, '.': 12}
    one_hot = np.zeros(8 * 8 * 13, dtype=np.float32)
    fen = fen.split()[0]
    row = 0
    col = 0
    for char in fen:
        if char.isdigit():
            col += int(char)
        elif char == '/':
            row += 1
            col = 0
        else:
            index = piece_to_index[char] + 13 * (row * 8 + col)
            one_hot[index] = 1
            col += 1
    return one_hot

def get_all_chess_data(datasets_dir=DATA_PATH, pathname='.txt'):
    all_results = []
    all_checkmates = []
    all_fens = []

    print("Files used to train:")
    for root, dirs, files in os.walk(datasets_dir):
        for file in files:
            if file.endswith(pathname):
                print(file)
                file_path = os.path.join(root, file)
                results, checkmates, fens = parse_chess_data(file_path)

                all_results.extend(results)
                all_checkmates.extend(checkmates)
                all_fens.extend(fens)
    return all_results, all_fens

def evaluer_echiquier(fen):
    valeurs_pieces = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
    score_blancs, score_noirs = 0, 0

    position_pieces = fen.split()[0]

    for piece in position_pieces:
        if piece.isalpha():
            if piece.isupper():
                score_blancs += valeurs_pieces[piece.lower()]
            else:
                score_noirs += valeurs_pieces[piece]

    return [score_blancs, score_noirs]


def test(mlp, test_data):
    correct_predictions = 0
    total_predictions = len(test_data)

    for x, y in test_data:
        prediction = mlp.predict(x)
        predicted_label = 1 if prediction[0, 0] > 0.5 else 0
        correct_predictions += (predicted_label == y[0])

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

def exec_torch(argv):
    if argv[1] == "-new":
        name_ia = argv[2]
        input_size = int(argv[3])
        num_neurons_first_layer = int(argv[4])
        num_hidden_layers = int(argv[5])
        neurons_per_hidden_layer = int(argv[6])
        mlp = MLP(name_ia=name_ia, input_size=input_size, num_neurons_first_layer=num_neurons_first_layer, num_hidden_layers=num_hidden_layers, neurons_per_hidden_layer=neurons_per_hidden_layer, output_neurons=1)
        mlp.print_info()
        mlp.save()
        return
    if argv[1] == "-train":
        mlp = load(argv[2])
        labels, training_inputs = get_all_chess_data(pathname=(argv[3]))
        training_data = [(np.reshape(x, (2, 1)), np.array([y])) for x, y in zip(training_inputs, labels)]
        print("\nWait...")
        mlp.train(training_data, int(argv[4]), mini_batch_size=1, eta=0.1)
        mlp.save()
        print("IA trained and saved!")
        return
    if argv[1] == "-test":
        mlp = load(argv[2])
        labels, training_inputs = get_all_chess_data(pathname=(argv[3]))
        training_data = [(np.reshape(x, (2, 1)), np.array([y])) for x, y in zip(training_inputs, labels)]
        accuracy = test(mlp, training_data)
        print(f"Précision du modèle: {accuracy:.2f}%")
