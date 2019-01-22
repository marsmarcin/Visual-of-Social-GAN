import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation

x=[12.66
,11.82
,10.98
,10.18
,9.41
,8.72
,7.99
,7.2
,6.39
,5.5
,4.65
,3.73
,2.76
,1.92
,1.01
,0.07]
y=[5.11
,4.85
,4.75
,4.55
,4.41
,4.26
,4.03
,3.96
,3.77
,3.53
,3.36
,3.12
,2.95
,2.82
,2.52
,2.2]
x1=[12.66
,11.82
,10.98
,10.18
,9.41
,8.72
,7.99
,7.2
,6.4357
,5.695
,4.9595
,4.228
,3.5009
,2.7791
,2.0645
,1.3587]
y1=[5.11
,4.85
,4.75
,4.55
,4.41
,4.26
,4.03
,3.96
,3.9484
,3.9436
,3.9499
,3.9656
,3.9911
,4.0265
,4.0716
,4.1256]
num_t=0
xdata, ydata = [], []
xdata1, ydata1 = [], []
fig, ax = plt.subplots()

#l = ax.plot(x, y,'.')
ln, = ax.plot([], [], 'r*')
ln1, = ax.plot([], [], 'b*')

def init():
    ax.set_xlim(-2, 15)
    ax.set_ylim(-2, 15)
    # return l

def gen_dot():
    for i in range(0,len(x1)):
        newdot = [x1[i], y1[i]]
        yield newdot

def update_dot(newd):
    xdata.append(newd[0])
    ydata.append(newd[1])
    global num_t

    xdata1.append(x[num_t])
    ydata1.append(y[num_t])
    num_t = num_t +1
    #dot.set_data(newd[0], newd[1])
    ln.set_data(xdata, ydata)
    ln1.set_data(xdata1, ydata1)
    return ln,ln1

ani = animation.FuncAnimation(fig, update_dot, frames = gen_dot, interval = 5, init_func=init)
# ani.save('sin_dot.gif', writer='imagemagick', fps=30)
# os.system("ffmpeg -i F://tracker_programe//sgan-masteranimation.gif")

plt.show()
