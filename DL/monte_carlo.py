from numpy import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import collections

def sigmoid_func(x):
    return 1/(1+np.exp(-x))


def figure_sigmoid_scatter(iterations):
    # 시그모이드 그리기 
    x = np.arange(-5, 5.1, 0.1)
    y = sigmoid_func(x)

    fig, ax = plt.subplots(figsize=(12,8))

    plt.plot(x, y, color = 'k')
    plt.grid(True)
    plt.axis([-6, 6, -0.1, 1.1])

    ax.add_patch(patches.Rectangle((-5,0), 10, 1, fill = False, linewidth = 2)) 
    
    # scatter 찍기
    random_x = random.uniform(-5, 5, size = iterations)
    random_y = random.uniform(0, 1, size = iterations)

    color = list(map(lambda x: 'blue' if x else 'red', random_y > sigmoid_func(random_x)))

    inner_ratio = collections.Counter(color)['red'] / len(color)

    plt.scatter(random_x, random_y, color = color, s = 2, label = f'area : {inner_ratio * 10}')
    plt.legend()

    plt.show()
    


if __name__ == '__main__':
    iterations = 100000

    figure_sigmoid_scatter(iterations)
