from pdb import set_trace
import numpy as np
import pygal

# def format(x):
#   return "(%s,%s)" % (x[0],x[1])

def helper(x_train, y_train, x_test, r, i=''):
    xy_chart = pygal.XY(stroke=False, range=(-r, r), xrange=(-r, r), x_title='経度', y_title='緯度')

    ones = [i for i, y in enumerate(y_train) if y == 1]
    zeros = [i for i, y in enumerate(y_train) if y == 0]

    xy_chart.title = '住所と何人暮らしかの関係'
    xy_chart.add('一人暮らし', x_train[ones], dots_size=4)
    xy_chart.add('ルームシェア', x_train[zeros], dots_size=4)
    xy_chart.add('Test Data', x_test, dots_size=4)
    xy_chart.render_to_file('scatter%s.svg' % i)
    xy_chart.render_in_browser()

def visualize1():
    x_train = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y_train = np.array([1,1,1,0,0,0])
    x_test = np.array([[1, 0], [-2, -2]])
    helper(x_train, y_train, x_test, 4)

def visualize2():
    x_train = np.array([[1, 1], [4, 4], [5, 5]])
    y_train = np.array([1,0,0])
    x_test = np.array([[0, 0]])
    helper(x_train, y_train, x_test, 5, 2)

visualize1()
visualize2()