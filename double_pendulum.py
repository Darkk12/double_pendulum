"""
===========================
The double pendulum problem
===========================

This animation illustrates the double pendulum problem.
"""

# Based on the example code at: https://matplotlib.org/examples/animation/double_pendulum_animated.html
# Edited for two pendulums by Jonas https://github.com/jonas37

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)

# th1(second pendulum), th2 (first pendulum), th1(second pendulum), th2 (second pendulum)
init_tuple = (120.0, 105.0, 120.0000001, 105.0)
#init_tuple = (180.0000001, 180.0, -180.0000001, 180.0)
#init_tuple = (180.0, 90.0000001, 180.0, 90.0)
#init_tuple = (80.0000001, 42.0, 80.0, 42.0)
#init_tuple = (120.0, -90.0, -120.0, 90.0)

th1 = init_tuple[0]
w1 = 0.0
th2 = init_tuple[1]
w2 = 0.0

th1_ = init_tuple[2]
w1_ = 0.0
th2_ = init_tuple[3]
w2_ = 0.0

# create a time array sampled at dt second steps
# duration given in seconds
duration = 100
dt = 0.02
t = np.arange(0, duration, dt)

#defines the number of samples fo which the drawn path should persist (None means forever)
PATH_RANGE = None


def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

    return dydx

# initial state
state = np.radians([th1, w1, th2, w2])
state_ = np.radians([th1_, w1_, th2_, w2_])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)
y_ = integrate.odeint(derivs, state_, t)

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])
x1_ = L1*sin(y_[:, 0])
y1_ = -L1*cos(y_[:, 0])

x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1
x2_ = L2*sin(y_[:, 2]) + x1_
y2_ = -L2*cos(y_[:, 2]) + y1_

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', color='C0', lw=2)
line_, = ax.plot([], [], 'o-', color='C1', lw=2)

path, = ax.plot([],[], color='C0', alpha=0.5)
path_, = ax.plot([],[], color='C1', alpha=0.5)

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
init_template = ("Init Conditions:\n"
                 "$\Theta_{11}:=%.7f\degree$ (blue)\n"
                 "$\Theta_{21}:=%.7f\degree$ (blue)\n"
                 "$\Theta_{12}:=%.7f\degree$ (orange)\n"
                 "$\Theta_{22}:=%.7f\degree$ (orange)"
                 %(th1,th2,th1_,th2_))
init_text = ax.text(0.05, 0.68, init_template, bbox=dict(boxstyle="round", fc="white"),size='smaller',transform=ax.transAxes)

pause = False
def onClick(event):
    global pause
    if pause:
        ani.event_source.start()
        pause = False
        #print("start")
    else:
        ani.event_source.stop()
        pause = True
        #print("stop")
        

def init():
    line.set_data([], [])
    line_.set_data([], [])
    
    path.set_data([], [])
    path_.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    thisx_ = [0, x1_[i], x2_[i]]
    thisy_ = [0, y1_[i], y2_[i]]
    
    start_range = i-PATH_RANGE if PATH_RANGE !=None else 0
    start_range = np.clip(start_range, 0, None)
    thispath = [x2[start_range:i], y2[start_range:i]]
    thispath_ = [x2_[start_range:i], y2_[start_range:i]]

    path.set_data(thispath)
    path_.set_data(thispath_)
    line.set_data(thisx, thisy)
    line_.set_data(thisx_, thisy_)
    time_text.set_text(time_template % (i*dt))
        
    return line, line_, path, path_, time_text

fig.canvas.mpl_connect('button_press_event', onClick)
ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
                              interval=(1/(len(t)/duration))*1000, blit=True, init_func=init)
plt.show()

# saving animation
# takes a whila and ffmpeg needs to be installed
#ani.save("double_pendulum_4.mp4")