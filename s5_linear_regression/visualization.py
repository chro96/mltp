import numpy as np
from bokeh.plotting import figure, output_file, show

x = np.array([0,1,2,3])
y = np.array([-1, 1, 3, 5])

output_file("plot/plot2.html")

p = figure(
   tools="pan,box_zoom,reset,save", title="linear regression example",
   x_axis_label='x', y_axis_label='y'
)

# p.line(x, y)
p.circle(x, y, fill_color="blue", size=8)
show(p)