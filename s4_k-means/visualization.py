from pdb import set_trace
import numpy as np
import pygal

X = np.array([[1,1],[1,2],[2,2],[4,5],[5,4]])

r = 5

def helper():
	chart = pygal.XY(stroke=False, range=(0, r), xrange=(0, r), x_title='経度', y_title='緯度')
	chart.title = 'Clustering'
	return chart

def visualize1():
	i = ''
	chart = helper()
	chart.add('X', X, dots_size=4)
	chart.render_to_file('scatter%s.svg' % i)
	chart.render_in_browser()

def visualize2():
	i = 2
	chart = helper()
	chart.add('cluster1', X[:1], dots_size=4)
	chart.add('cluster2', X[1:], dots_size=4)
	chart.add('cluster centers', np.array([[1.05,1],[1.05,2]]), dots_size=4)
	chart.render_to_file('scatter%s.svg' % i)
	chart.render_in_browser()

def visualize3():
	i = 3
	chart = helper()
	chart.add('cluster1', X[:3], dots_size=4)
	chart.add('cluster2', X[3:], dots_size=4)
	chart.add('cluster centers', np.array([[1.05,1],[3,3.25]]), dots_size=4)
	chart.render_to_file('scatter%s.svg' % i)
	chart.render_in_browser()

def visualize4():
	i = 4
	chart = helper()
	chart.add('cluster1', X[:3], dots_size=4)
	chart.add('cluster2', X[3:], dots_size=4)
	chart.add('cluster centers', np.array([[1.333,1.667],[4.5,4.5]]), dots_size=4)
	chart.render_to_file('scatter%s.svg' % i)
	chart.render_in_browser()

visualize1()
visualize2()
visualize3()
visualize4()

line_chart = pygal.Line()
line_chart.title = 'クラスター数とinertiaの関係'
line_chart.x_labels = ['2','3','4','5','6']
line_chart.add('inertia',  [14.2, 5,4,3.5,3.2])
line_chart.render_to_file('inertia.svg')
line_chart.render_in_browser()