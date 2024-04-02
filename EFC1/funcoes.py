""""
Contem os codigos utilizados em main.py
"""


def treino(y, num, k):
    """"
    y:      dados de treinamento para os tempos 'x's
    num:    tamanho do array de 'x'
    k:      numero de pontos passados
    """

    train_x = []
    train_y = []

    for i in range(num):

        # if i == 100:
        #     print('pare aqui')
        if num - k - i - 1 < 0 & num - i - 1 >= 0:
            train_x.append([0] * (i + k + 1 - num) + y[0:num - i - 1])
        else:
            train_x.append(y[num - k - i - 1:num - i - 1])
        train_x[i].reverse()
        train_y.append(y[num - 1 - i])

    train_x.reverse()
    train_y.reverse()

    del train_x[0]
    del train_y[0]
    return train_x, train_y


def validacao(y_test, y, num, k):
    """"
    y_test: dados do treinamento utilizados para a validacao
    y:      dados de validacao para os tempos 'x's
    num:    tamanho do array de validacao
    k:      numero de pontos passados
    """

    valid_x = []
    valid_y = []

    for i in range(num):

        # if i == 100:
        #     print('pare aqui')
        if i < k:
            valid_x.append(y_test[-k + i:] + y[:i])
        elif i == k:
            valid_x.append(y[:i])
        else:
            valid_x.append(y[i - k:i])
        valid_x[i].reverse()
        valid_y.append(y[i])

    return valid_x, valid_y
