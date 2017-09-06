import os
import numpy as np
import pickle

def scripts_import(path_to_files):
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

    X_true = np.zeros((len(chars_lookup), 1))
    X_false = np.zeros((len(chars_lookup), 1))
    Y_true = np.zeros((1, 1))
    Y_false = np.zeros((1, 1))

    for root, dirs, files in os.walk(path_to_files):
        for name in files:
            if name[-4] == ".png" or name[-4] == ".jpg": continue
            temp_vec_x = np.zeros((len(chars_lookup), 1))
            script = open(os.path.join(root, name))
            lines = list(script.readlines())
            number_of_lines = len(lines)
            for line in lines:
                if line.strip() == '' or line == '\n':
                    continue
                for char in line:
                    try:
                        temp_vec_x[chars_lookup[char.lower()], 0] += 1 if char != '\n' else 0
                    except KeyError:
                        number_of_lines -= 1
                        break
            if number_of_lines <= 0: continue
            if name[-5:] == ".html" or name[-4:] == ".htm":
                X_true = np.concatenate((X_true, temp_vec_x/number_of_lines), axis=1)
                Y_true = np.concatenate((Y_true, np.array([[1]])), axis=1)
            else:
                X_false = np.concatenate((X_false, temp_vec_x/number_of_lines), axis=1)
                Y_false = np.concatenate((Y_false, np.array([[0]])), axis=1)
            script.close()
    return X_true[:, 1:], Y_true[:, 1:], X_false[:, 1:], Y_false[:, 1:]


XT, YT, XF, YF = scripts_import("data/html_vs_non_html")

fileObject = open('XT_gist.pkl','wb')
pickle.dump(XT,fileObject)
fileObject.close()

fileObject = open('XF_gist.pkl','wb')
pickle.dump(XF,fileObject)
fileObject.close()

fileObject = open('YT_gist.pkl','wb')
pickle.dump(YT, fileObject)
fileObject.close()

fileObject = open('YF_gist.pkl','wb')
pickle.dump(YF, fileObject)
fileObject.close()
