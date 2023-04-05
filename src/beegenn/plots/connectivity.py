import numpy as np

def plot_connectivity(connectivity_matrix, first_repeat, second_repeat, pop, subplot):
    print(first_repeat//connectivity_matrix.shape[0])
    print(second_repeat//connectivity_matrix.shape[0])
    matrix = np.repeat(connectivity_matrix, repeats = first_repeat//connectivity_matrix.shape[0], axis = 0)
    matrix = np.repeat(matrix, repeats = second_repeat//connectivity_matrix.shape[1], axis = 1)
    print(matrix.shape)
    res = subplot.imshow(matrix, cmap = 'plasma')
    subplot.set_title(pop)

    return res
