import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

EPSILON = 0.001 

class ThreeBodySim:
    def __init__(self):
        self.masses = np.array([10.0, 10.0, 10.0])
        self.G = 1.0

        self.dt = 0.01

        self.pos = np.array([
            [1.0, 0.0],    # r1
            [-0.5, 0.866], # r2
            [-0.5, -0.866] # r3
        ], dtype='float64')

       # self.vel = np.array([
        #    [0.2, 0.2],
        #    [-0.2, 0.2],
        #    [0.0, -0.4]
       # ], dtype='float64')

        self.vel = np.array([
            [0, 0],
            [0, 0],
            [0, 0]
        ], dtype='float64')

        self.history_len = 50
        self.history = [np.zeros((self.history_len, 2)) for _ in range(3)]

    def compute_accelerations(self):

        r1, r2, r3 = self.pos[0], self.pos[1], self.pos[2]
        m1, m2, m3 = self.masses[0], self.masses[1], self.masses[2]

        r1_r2 = r1 - r2
        r1_r3 = r1 - r3
        r2_r3 = r2 - r3
        r2_r1 = r2 - r1
        r3_r1 = r3 - r1
        r3_r2 = r3 - r2

        dist_r1_r2 = np.linalg.norm(r1_r2)
        dist_r1_r3 = np.linalg.norm(r1_r3)
        dist_r2_r3 = np.linalg.norm(r2_r3)
        dist_r2_r1 = np.linalg.norm(r2_r1)
        dist_r3_r1 = np.linalg.norm(r3_r1)
        dist_r3_r2 = np.linalg.norm(r3_r2)


        acc = np.zeros_like(self.pos)

        # RÓWNANIE 1: r1_bis = -G*m2*(r1-r2)/|r1-r2|^3 - G*m3*(r1-r3)/|r1-r3|^3
        acc[0] = -self.G * m2 * (r1_r2) / (dist_r1_r2**3 + EPSILON) \
                 -self.G * m3 * (r1_r3) / (dist_r1_r3**3 + EPSILON)

        # RÓWNANIE 2: r2_bis = -G*m3*(r2-r3)/|r2-r3|^3 - G*m1*(r2-r1)/|r2-r1|^3
        acc[1] = -self.G * m3 * (r2_r3) / (dist_r2_r3**3 + EPSILON) \
                 -self.G * m1 * (r2_r1) / (dist_r2_r1**3 + EPSILON)

        # RÓWNANIE 3: r3_bis = -G*m1*(r3-r1)/|r3-r1|^3 - G*m2*(r3-r2)/|r3-r2|^3
        acc[2] = -self.G * m1 * (r3_r1) / (dist_r3_r1**3 + EPSILON) \
                 -self.G * m2 * (r3_r2) / (dist_r3_r2**3 + EPSILON)

        return acc

    def update(self):
        """Metoda całkowania numerycznego."""
        acc = self.compute_accelerations()
        

        self.vel += acc * self.dt

        self.pos += self.vel * self.dt


        for i in range(3):
            self.history[i] = np.roll(self.history[i], -1, axis=0)
            self.history[i][-1] = self.pos[i]

sim = ThreeBodySim()

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.35)

ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title("Symulacja wg równań Newtona")

colors = ['red', 'green', 'blue']
bodies_plot = [ax.plot([], [], 'o', color=c, ms=12, label=f'Ciało {i+1}')[0] for i, c in enumerate(colors)]
trails_plot = [ax.plot([], [], '-', color=c, alpha=0.5, lw=1)[0] for c in colors]
ax.legend(loc='upper right')

def animate(frame):
    sim.update()
    for i in range(3):
        bodies_plot[i].set_data([sim.pos[i, 0]], [sim.pos[i, 1]])
        trails_plot[i].set_data(sim.history[i][:, 0], sim.history[i][:, 1])
    return bodies_plot + trails_plot


ax_G = plt.axes([0.2, 0.30, 0.65, 0.03])
s_G = Slider(ax_G, 'G', -10, 10.0, valinit=sim.G)
ax_dt = plt.axes([0.2, 0.25, 0.65, 0.03])
ax_m1 = plt.axes([0.2, 0.20, 0.65, 0.03])
ax_m2 = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_m3 = plt.axes([0.2, 0.10, 0.65, 0.03])

s_dt = Slider(ax_dt, 'Delta t', 0.001, 0.05, valinit=sim.dt)
s_m1 = Slider(ax_m1, 'm1 (Czerwona)', 1.0, 50.0, valinit=sim.masses[0])
s_m2 = Slider(ax_m2, 'm2 (Zielona)', 1.0, 50.0, valinit=sim.masses[1])
s_m3 = Slider(ax_m3, 'm3 (Niebieska)', 1.0, 50.0, valinit=sim.masses[2])

def update_params(val):
    sim.dt = s_dt.val
    sim.masses[0] = s_m1.val
    sim.masses[1] = s_m2.val
    sim.masses[2] = s_m3.val

s_dt.on_changed(update_params)
s_m1.on_changed(update_params)
s_m2.on_changed(update_params)
s_m3.on_changed(update_params)

ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_reset = Button(ax_reset, 'Reset', hovercolor='0.975')

def reset(event):
    sim.pos = np.array([[1.0, 0.0], [-0.5, 0.866], [-0.5, -0.866]], dtype='float64')
    sim.vel = np.array([[0.2, 0.2], [-0.2, 0.2], [0.0, -0.4]], dtype='float64')
    sim.history = [np.zeros((sim.history_len, 2)) for _ in range(3)]
    s_m1.reset()
    s_m2.reset()
    s_m3.reset()

btn_reset.on_clicked(reset)

ani = FuncAnimation(fig, animate, frames=200, interval=20, blit=True)
plt.show()