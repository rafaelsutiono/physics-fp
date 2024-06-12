import easygui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def basketball_shot_simulation():
    # set hoop height and distance
    hoop_msg = "Enter hoop height and distance from starting point"
    hoop_title = "Input for hoop position"
    hoop_fields = ["Hoop Height (m)", "Hoop Distance (m)"]
    hoop_values = easygui.multenterbox(hoop_msg, hoop_title, hoop_fields)
    if hoop_values is None:
        return False
    hoop_height, hoop_distance = float(hoop_values[0]), float(hoop_values[1])

    # hoop size
    hoop_radius = 0.23  # Radius of the hoop in meters

    g = 9.807  # value of gravity (m/s^2)
    weight = 0.625  # weight of standard basketball in kg
    drag_coefficient = 0.47  # typical drag value for a spherical object
    air_density = 1.225  # kg/m^3 (density of air at sea level)
    cross_sectional_area = 0.0453  # m^2 (cross-sectional area of a standard basketball)
    restitution_coefficient = 0.75  # coefficient of restitution for a basketball
    initial_height = 2.0  # avg initial height from which a basketball is thrown

    def equations(t, y):
        vx, vy = y[2], y[3]
        speed = np.sqrt(vx**2 + vy**2)
        drag_force = 0.5 * drag_coefficient * air_density * cross_sectional_area * speed**2
        drag_acceleration = drag_force / weight
        ax = -drag_acceleration * (vx / speed)
        ay = -g - drag_acceleration * (vy / speed)
        return [vx, vy, ax, ay]

    def simulate_bounce(y0, t_max=10, dt=0.01):
        t = 0
        positions = []
        while t < t_max:
            t_span = (t, t + dt)
            sol = solve_ivp(equations, t_span, y0, t_eval=[t + dt], rtol=1e-8, atol=1e-8)
            y0 = sol.y[:, -1]
            positions.append(y0)
            t += dt
            if y0[1] <= 0:  # when it hits the ground
                y0[1] = 0  # reset height to ground level
                y0[3] = -y0[3] * restitution_coefficient  # reverse and reduce vertical velocity
                if abs(y0[3]) < 0.1:  # so that bouncing evventually stops as velocity decreases
                    break
        return np.array(positions)

    ## objective function to minimize distance from the ball to the hoop ##
    def objective(vars):
        v, theta_deg = vars
        theta = theta_deg * np.pi / 180.0  # convert angle to radians
        vx0 = v * np.cos(theta)
        vy0 = v * np.sin(theta)
        y0 = [0, initial_height, vx0, vy0]
        positions = simulate_bounce(y0)
        x_points = positions[:, 0]
        y_points = positions[:, 1]
        
        # calculate the distance to the hoop
        distances = np.sqrt((x_points - hoop_distance)**2 + (y_points - hoop_height)**2)
        min_distance = np.min(distances)
        return min_distance

    # find the optimal initial velocity and angle
    initial_guess = [10, 45]  # initial guess for the optimization [velocity (m/s), angle (degrees)]
    result = minimize(objective, initial_guess, bounds=[(0, 50), (0, 90)])
    optimal_velocity, optimal_angle = result.x

    # use the optimal velocity and angle for the simulation
    v = optimal_velocity
    theta = optimal_angle * np.pi / 180.0  # Convert angle to radians
    vx0 = v * np.cos(theta)
    vy0 = v * np.sin(theta)
    y0 = [0, initial_height, vx0, vy0]

    # simulate the projectile motion with the optimal values
    positions = simulate_bounce(y0)
    x_points = positions[:, 0]
    y_points = positions[:, 1]

    # adjust axes limits based on computed range and height
    x_max = max(x_points[-1], hoop_distance) + 5  # margin for better visualization
    y_max = max(max(y_points), hoop_height) + 5  # margin for better visualization

    # animation
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    # basketball png
    basketball_img = plt.imread("basketballll.png")
    imagebox = OffsetImage(basketball_img, zoom=0.025)  # Adjust the zoom level as needed

    # label key points
    time_text = f"Flight Time: {round(positions[-1, 0], 2)}s"
    h_point = f"Highest Point: {round(max(y_points), 2)}m"
    range_projectile = f"Range: {round(x_points[-1], 2)}m"
    plt.plot(x_points[-1], 0, 'go', label="Landing Point")
    plt.plot(x_points[np.argmax(y_points)], max(y_points), 'ro', label="Highest Point")

    # adjust the text placement for better visibility
    plt.text(x_points[np.argmax(y_points)] + 1, max(y_points) - 2, h_point, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(x_points[-1] - 10, -3, range_projectile, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(x_points[np.argmax(y_points)] + 1, max(y_points) + 2.5, time_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # display optimal velocity and angle on the graph
    optimal_text = f"Optimal Velocity: {round(optimal_velocity, 2)} m/s\nOptimal Angle: {round(optimal_angle, 2)}°"
    plt.text(x_max * 0.05, y_max * 0.95, optimal_text, fontsize=14, bbox=dict(facecolor='lightblue', alpha=0.7))

    # load hoop png
    hoop_img = plt.imread("hoopp.png")
    hoop_imagebox = OffsetImage(hoop_img, zoom=0.05)  # adjust size

    # AnnotationBbox to display hoop png
    hoop_ab = AnnotationBbox(hoop_imagebox, (hoop_distance, hoop_height), frameon=False)
    ax.add_artist(hoop_ab)

    line, = ax.plot([], [], lw=2.5, color='b')  # initialize the motion line
    ab = AnnotationBbox(imagebox, (0, 0), frameon=False)  # initialize the annotation box with the basketball png
    ax.add_artist(ab)

    total_frames = len(x_points) + 50  # delay for one second before looping animation

    def init():
        line.set_data([], [])
        ab.xybox = (0, 0)  # initial position of basketball png
        return line, ab,

    def animate(i):
        if i < len(x_points):
            line.set_data(x_points[:i], y_points[:i])
            ab.xybox = (x_points[i], y_points[i])  # update position of basketball png
        elif i == total_frames - 1:
            return init()
        return line, ab,

    plt.axhline(0, color='gray')
    plt.axvline(0, color='gray')
    plt.xlabel('Distance (m)', fontsize=14)
    plt.ylabel('Height (m)', fontsize=14)
    plt.title('Basketball Shot Simulation', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=total_frames, interval=20, blit=True, repeat=True)
    plt.show()

    retry = easygui.ynbox('Would you like to try again with different settings?', 'Retry?', ('Yes', 'No'))
    return retry