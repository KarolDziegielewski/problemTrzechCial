import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons

EPSILON = 0.05 

class ThreeBodySim:
    def __init__(self):
        self.history_len = 150
        self.dt = 0.01
        self.G = self.K = 1.5
        self.masses = np.array([15.0, 10.0, 12.0])
        self.charges = np.array([2.0, -2.0, 1.0])
        self.mode = 'Grawit.'
        self.is_running = False
        self.reset_positions()

    def reset_positions(self):
        self.pos = np.array([[1.2, 0.2], [-0.8, 1.0], [-0.4, -1.2]], dtype='float64')
        self.vel = np.array([[-0.2, -0.1], [0.3, 0.2], [-0.1, -0.4]], dtype='float64')
        self.history = [np.full((self.history_len, 2), p) for p in self.pos]

    def compute_accelerations(self):
        acc = np.zeros_like(self.pos)
        for i in range(3):
            for j in range(3):
                if i == j: continue
                r_vec = self.pos[j] - self.pos[i]
                dist = np.linalg.norm(r_vec)
                if self.mode == 'Grawit.':
                    mag = self.G * self.masses[j] / (dist**3 + EPSILON)
                else:
                    mag = -self.K * (self.charges[i] * self.charges[j]) / (self.masses[i] * (dist**3 + EPSILON))
                acc[i] += mag * r_vec
        return acc

    def update(self):
        if not self.is_running:
            return
        a = self.compute_accelerations()
        self.vel += a * self.dt
        self.pos += self.vel * self.dt
        for i in range(3):
            self.history[i] = np.roll(self.history[i], -1, axis=0)
            self.history[i][-1] = self.pos[i]

sim = ThreeBodySim()
sim.is_running = False

fig = plt.figure(figsize=(10, 10))
ax = fig.add_axes([0.1, 0.35, 0.8, 0.6])
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)

colors = ['#FF3333', '#33FF33', '#3333FF']
bodies_p = [ax.plot([], [], 'o', color=colors[i], markeredgecolor='black', zorder=5)[0] for i in range(3)]
trails_p = [ax.plot([], [], '-', color=colors[i], alpha=0.4, lw=2, zorder=4)[0] for i in range(3)]

ax_m1 = fig.add_axes([0.15, 0.22, 0.25, 0.02]); s_m1 = Slider(ax_m1, 'Masa R ', 1, 100, valinit=15.0)
ax_m2 = fig.add_axes([0.15, 0.18, 0.25, 0.02]); s_m2 = Slider(ax_m2, 'Masa G ', 1, 100, valinit=10.0)
ax_m3 = fig.add_axes([0.15, 0.14, 0.25, 0.02]); s_m3 = Slider(ax_m3, 'Masa B ', 1, 100, valinit=12.0)
ax_q1 = fig.add_axes([0.6, 0.22, 0.25, 0.02]); s_q1 = Slider(ax_q1, 'Ład. R ', -10, 10, valinit=2.0)
ax_q2 = fig.add_axes([0.6, 0.18, 0.25, 0.02]); s_q2 = Slider(ax_q2, 'Ład. G ', -10, 10, valinit=-2.0)
ax_q3 = fig.add_axes([0.6, 0.14, 0.25, 0.02]); s_q3 = Slider(ax_q3, 'Ład. B ', -10, 10, valinit=1.0)
ax_G = fig.add_axes([0.15, 0.08, 0.25, 0.02]); s_G = Slider(ax_G, 'Stała ', 0.1, 10.0, valinit=1.5)
ax_dt = fig.add_axes([0.6, 0.08, 0.25, 0.02]); s_dt = Slider(ax_dt, 'Krok t ', 0.001, 0.05, valinit=0.01)

def update_all(_):
    sim.masses = np.array([s_m1.val, s_m2.val, s_m3.val])
    sim.charges = np.array([s_q1.val, s_q2.val, s_q3.val])
    sim.G = sim.K = s_G.val
    sim.dt = s_dt.val

sliders_list = [s_m1, s_m2, s_m3, s_q1, s_q2, s_q3, s_G, s_dt]
for s in sliders_list:
    s.on_changed(update_all)

rax = fig.add_axes([0.02, 0.7, 0.12, 0.08], facecolor='#f0f0f0')
radio = RadioButtons(rax, ('Grawit.', 'Coulomb'))
radio.set_active(0)

def change_mode(label):
    sim.mode = label
    if sim.mode == 'Grawit.':
        ax_start.buttoncolor = '#4CAF50'
        btn_start.color = '#4CAF50'
    else:
        ax_start.buttoncolor = '#2196F3'
        btn_start.color = '#2196F3'
    fig.canvas.draw_idle()

radio.on_clicked(change_mode)

def start_stop_sim(_):
    sim.is_running = not sim.is_running
    if sim.is_running:
        btn_start.label.set_text('STOP')
        btn_start.color = '#FF5722'
        btn_start.hovercolor = '#FF8A65'
        update_all(None)
        sim.mode = radio.value_selected
    else:
        btn_start.label.set_text('START')
        if sim.mode == 'Grawit.':
            btn_start.color = '#4CAF50'
        else:
            btn_start.color = '#2196F3'
        btn_start.hovercolor = '#A5D6A7'
    fig.canvas.draw_idle()

def reset_sim(_):
    sim.is_running = False
    btn_start.label.set_text('START')
    if sim.mode == 'Grawit.':
        btn_start.color = '#4CAF50'
    else:
        btn_start.color = '#2196F3'
    btn_start.hovercolor = '#A5D6A7'
    
    for s in sliders_list:
        s.eventson = False
        s.reset()
        s.eventson = True
    
    radio.eventson = False
    radio.set_active(0)
    radio.eventson = True
    
    update_all(None)
    
    sim.mode = 'Grawit.'
    
    sim.reset_positions()
    
    for i in range(3):
        trails_p[i].set_data([], [])
        bodies_p[i].set_data([], [])
    
    fig.canvas.draw_idle()

ax_start = fig.add_axes([0.15, 0.02, 0.25, 0.04])
btn_start = Button(ax_start, 'URUCHOM SYMULACJĘ', color='#4CAF50', hovercolor='#A5D6A7')
btn_start.on_clicked(start_stop_sim)

ax_reset = fig.add_axes([0.6, 0.02, 0.25, 0.04])
btn_reset = Button(ax_reset, 'ZRESETUJ', color='#757575', hovercolor='#BDBDBD')
btn_reset.on_clicked(reset_sim)

def animate(_):
    sim.update()
    for j in range(3):
        bodies_p[j].set_data([sim.pos[j,0]], [sim.pos[j,1]])
        bodies_p[j].set_markersize(5 + np.sqrt(sim.masses[j])*2)
        trails_p[j].set_data(sim.history[j][:,0], sim.history[j][:,1])
    return bodies_p + trails_p

update_all(None)
sim.mode = 'Grawit.'

ani = FuncAnimation(fig, animate, interval=15, blit=True, cache_frame_data=False)

plt.show()
