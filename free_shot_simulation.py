import easygui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from scipy.integrate import solve_ivp


def free_shot_simulation():
    msg = "Enter velocity (m/s), angle (degrees), and weight (kg)"
    title = "Input for projectile simulation"
    fieldNames = ["Velocity (m/s)", "Angle (degrees)", "Weight (kg)"]
    fieldValues = []  # initialize array for values
    fieldValues = easygui.multenterbox(msg, title, fieldNames)
    print(fieldValues)

    while 1:
        if fieldValues is None:
            break
        errmsg = ""
        for i in range(len(fieldNames)):
            if fieldValues[i].strip() == "":
                errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
        if errmsg == "":
            break 
        fieldValues = easygui.multenterbox(errmsg, title, fieldNames, fieldValues)
    print("Reply was:", fieldValues)

    # animation
    sns.set()
    fig, ax = plt.subplots()
    g = 9.807  # value of gravity (m/s^2)
    v = float(fieldValues[0])  # initial velocity (m/s)
    theta = float(fieldValues[1]) * np.pi / 180.0  # initial angle of launch in radians
    weight = float(fieldValues[2])  # weight of the projectile in kg
    drag_coefficient = 0.47  # typical value for a spherical object
    air_density = 1.225  # kg/m^3 (density of air at sea level)
    cross_sectional_area = 0.045  # m^2 (assumed cross-sectional area of the projectile)
    restitution_coefficient = 0.6  # coefficient of restitution for bounces

    # function to calculate the motion under gravity and drag
    def equations(t, y):
        vx, vy = y[2], y[3]
        speed = np.sqrt(vx**2 + vy**2)
        drag_force = 0.5 * drag_coefficient * air_density * cross_sectional_area * speed**2
        drag_acceleration = drag_force / weight
        ax = -drag_acceleration * (vx / speed)
        ay = -g - drag_acceleration * (vy / speed)
        return [vx, vy, ax, ay]

    # initial conditions
    vx0 = v * np.cos(theta)
    vy0 = v * np.sin(theta)
    y0 = [0, 0, vx0, vy0]

    # function to simulate projectile motion with realistic bounces
    def simulate_bounce(y0, t_max=10, dt=0.01):
        t = 0
        positions = []
        while t < t_max:
            t_span = (t, t + dt)
            sol = solve_ivp(equations, t_span, y0, t_eval=[t + dt], rtol=1e-8, atol=1e-8)
            y0 = sol.y[:, -1]
            positions.append(y0)
            t += dt
            if y0[1] <= 0: 
                y0[1] = 0  
                y0[2] *= restitution_coefficient  # reduce horizontal velocity due to energy loss
                y0[3] = -y0[3] * restitution_coefficient  # reverse and reduce vertical velocity
                if np.sqrt(y0[2]**2 + y0[3]**2) < 0.1:  # stop bouncing if the total velocity is very small
                    break
        return np.array(positions)

    # initial conditions
    vx0 = v * np.cos(theta)
    vy0 = v * np.sin(theta)
    y0 = [0, 0, vx0, vy0]

    # simulate the projectile motion
    positions = simulate_bounce(y0)
    x_points = positions[:, 0]
    y_points = positions[:, 1]

    # label key points
    time_text = "Flight Time: " + str(round(positions[-1, 0], 2)) + 's'
    h_point = "Highest Point: " + str(round(max(y_points), 2)) + 'm'
    range_projectile = "Range: " + str(round(x_points[-1], 2)) + 'm'
    plt.plot(x_points[-1], 0, 'go')
    plt.plot(x_points[np.argmax(y_points)], max(y_points), 'ro')

    # text placement for better visibility
    plt.text(x_points[np.argmax(y_points)] + 1, max(y_points) - 2, h_point, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(x_points[-1] - 10, -3, range_projectile, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(x_points[np.argmax(y_points)] + 1, max(y_points) + 2.5, time_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # adjust axes limits based on computed range and height
    x_max = x_points[-1] + x_points[-1] * 0.1
    y_max = max(y_points) + max(y_points) * 0.1 

    line, = ax.plot([], [], lw=3.5)  # initialize the line
    total_frames = len(x_points) + 50  # delay after looping

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        if i < len(x_points):
            line.set_data(x_points[:i], y_points[:i])
        elif i == total_frames - 1:
            return init()
        return line,

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.xlabel('Distance (m)')
    plt.ylabel('Distance (m)')
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=total_frames, interval=20, blit=True, repeat=True)
    plt.show()

    retry = easygui.ynbox('Would you like to try again with different settings?', 'Retry?', ('Yes', 'No'))
    return retry
