from pdb import set_trace
import numpy as np
from bokeh.plotting import figure, output_file, show

def helper(x_train, y_train, x_test, i=''):
    output_file("plot/scatter%s.html" % i)

    p = figure(
       tools="pan,box_zoom,reset,save", title="住所と何人暮らしかの関係",
       x_axis_label='経度', y_axis_label='緯度'
    )

    ones = x_train[np.where(y_train == 1)]
    p.circle(ones.T[0], ones.T[1], fill_color="red", size=8)
    zeros = x_train[np.where(y_train == 0)]
    p.circle(zeros.T[0], zeros.T[1], fill_color="blue", size=8)
    p.circle(x_test.T[0], x_test.T[1], fill_color="green", size=8)
    show(p)

def visualize1():
    x_train = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y_train = np.array([1,1,1,0,0,0])
    x_test = np.array([[1, 0], [-2, -2]])
    helper(x_train, y_train, x_test)

def visualize2():
    x_train = np.array([[1, 1], [4, 4], [5, 5]])
    y_train = np.array([1,0,0])
    x_test = np.array([[0, 0]])
    helper(x_train, y_train, x_test, 2)

visualize1()
visualize2()



