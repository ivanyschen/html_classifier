import os
import numpy as np

def file_to_lines(path, file_name):
    file = open(os.path.join(path, file_name), 'r')
    return file.readlines()
def lines_to_vec(lines):
    chars_lookup = {
        'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5,
        'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11,
        'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17,
        's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23,
        'y': 24, 'z': 25, '!': 26, '#': 27, '"': 28, "'": 29,
        '-': 30, ',': 31, '/': 32, '.': 33, ':': 34, '=': 35,
        '<': 36, '?': 37, '>': 38, '0': 39, '1': 40, '2': 41,
        '3': 42, '4': 43, '5': 44, '6': 45, '7': 46, '8': 47,
        '9': 48, ' ': 49, '(': 50, ')': 51, '+': 52, '%': 53,
        '$': 54, '&': 55, '|': 56, ';': 57, '*': 58, '@': 59,
        '_': 60, '{': 61, '}': 62, '\\': 63, '[': 64, ']': 65
    }
    data_x_true = np.zeros((len(chars_lookup), 1))
    data_x_false = np.zeros((len(chars_lookup), 1))

    data_y_true = np.zeros((1, 1))
    data_y_false = np.zeros((1, 1))

    keywords = set(["user", "email", "e-mail", "password", "sign in", "signin"])
    for line in lines:
        temp_vec_x = np.zeros((len(chars_lookup), 1))
        if line.strip() == '':
            continue
        for char in line.strip():
            if char !='\n':
                temp_vec_x[chars_lookup[char.lower()], 0] += 1
        has_keyword = False
        for k in keywords:
            if k in line.lower():
                data_x_true = np.concatenate((data_x_true, temp_vec_x), axis=1)
                data_y_true = np.concatenate((data_y_true, np.array([[1]])), axis=1)
                has_keyword = True
                break
        if not has_keyword:
            data_x_false = np.concatenate((data_x_false, temp_vec_x), axis=1)
            data_y_false = np.concatenate((data_y_false, np.array([[0]])), axis=1)

    return data_x_true[:, 1:], data_y_true[:, 1:], data_x_false[:, 1:], data_y_false[:, 1:]

lines = file_to_lines("data/htmls", "1.html")
X_t, Y_t, X_f, Y_f = lines_to_vec(lines)