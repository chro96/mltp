from pdb import set_trace
import numpy as np
from bokeh.plotting import figure, output_file, show

X = np.array([[1,1],[1,2],[2,2],[4,5],[5,4]])

colors = ['red','blue','green']

def helper(data, i=''):
    output_file("plot/scatter%s.html" % i)

    p = figure(
       tools="pan,box_zoom,reset,save", title="Clustering",
       x_axis_label='経度', y_axis_label='緯度'
    )

    for i, d in enumerate(data):
        p.circle(d.T[0], d.T[1], fill_color=colors[i], size=8)
    show(p)

helper([X])
helper([X[:1], X[1:], np.array([[1.05,1],[1.05,2]])], 2)
helper([X[:3], X[3:], np.array([[1.05,1],[3,3.25]])], 3)
helper([X[:3], X[3:], np.array([[1.333,1.667],[4.5,4.5]])], 4)

output_file("plot/inertia.html")
p = figure(
   tools="pan,box_zoom,reset,save", title="Clustering",
   x_axis_label='クラスター数', y_axis_label='inertia'
)
x = [2,3,4,5,6]
y = [14.2, 5,4,3.5,3.2]
p.line(x, y)
p.circle(x, y)
show(p)



