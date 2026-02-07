import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons

# Zapobiega "wystrzeliwaniu" ciał przy bliskich spotkaniach
EPSILON = 0.05 

class ThreeBodySim:
    def __init__(self):
        # Parametry fizyczne
        self.masses = np.array([15.0, 10.0, 12.0])
        self.charges = np.array([2.0, -2.0, 1.0])
        self.G = 1.5
        self.K = 1.5
        self.dt = 0.01
        self.mode = 'Grawitacja'

        # Warunki początkowe - asymetryczne dla ciekawszego ruchu
        self.pos = np.array([
            [1.2, 0.2],    # Ciało 1
            [-0.8, 1.0],   # Ciało 2
            [-0.4, -1.2]   # Ciało 3
        ], dtype='float64')

        self.vel = np.array([
            [-0.2, -0.1], 
            [0.3, 0.2],   
            [-0.1, -0.4]
        ], dtype='float64')

        # Historia śladu
        self.history_len = 150
        self.history = [np.zeros((self.history_len, 2)) for _ in range(3)]
        for i in range(3):
            self.history[i][:] = self.pos[i]

    def compute_accelerations(self):
        acc = np.zeros_like(self.pos)
        for i in range(3):
            for j in range(3):
                if i == j: continue
                r_vec = self.pos[j] - self.pos[i]
                dist = np.linalg.norm(r_vec)
                
                if self.mode == 'Grawitacja':
                    # Newton: siła zawsze przyciąga
                    mag = self.G * self.masses[j] / (dist**3 + EPSILON)
                else:
                    # Coulomb: kierunek zależy od znaków ładunków
                    mag = -self.K * (self.charges[i] * self.charges[j]) / (self.masses[i] * (dist**3 + EPSILON))
                
                acc[i] += mag * r_vec
        return acc

    def update(self):
        # Schemat Eulera-Cromera
        a = self.compute_accelerations()
        self.vel += a * self.dt
        self.pos += self.vel * self.dt
        
        for i in range(3):
            self.history[i] = np.roll(self.history[i], -1, axis=0)
            self.history[i][-1] = self.pos[i]

sim = ThreeBodySim()
fig = plt.figure(figsize=(10, 10))
ax = fig.add_axes([0.1, 0.35, 0.8, 0.6])

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)

colors = ['#FF3333', '#33FF33', '#3333FF']
bodies_p = [ax.plot([], [], 'o', color=colors[i], markeredgecolor='black', zorder=5)[0] for i in range(3)]
trails_p = [ax.plot([], [], '-', color=colors[i], alpha=0.4, lw=2, zorder=4)[0] for i in range(3)]


ax_m1 = fig.add_axes([0.15, 0.22, 0.25, 0.02])
ax_m2 = fig.add_axes([0.15, 0.18, 0.25, 0.02])
ax_m3 = fig.add_axes([0.15, 0.14, 0.25, 0.02])

ax_q1 = fig.add_axes([0.6, 0.22, 0.25, 0.02])
ax_q2 = fig.add_axes([0.6, 0.18, 0.25, 0.02])
ax_q3 = fig.add_axes([0.6, 0.14, 0.25, 0.02])

ax_G = fig.add_axes([0.15, 0.08, 0.25, 0.02])
ax_dt = fig.add_axes([0.6, 0.08, 0.25, 0.02])

s_m1 = Slider(ax_m1, 'Masa R ', 1, 100, valinit=sim.masses[0])
s_m2 = Slider(ax_m2, 'Masa G ', 1, 100, valinit=sim.masses[1])
s_m3 = Slider(ax_m3, 'Masa B ', 1, 100, valinit=sim.masses[2])

s_q1 = Slider(ax_q1, 'Ład. R ', -10, 10, valinit=sim.charges[0])
s_q2 = Slider(ax_q2, 'Ład. G ', -10, 10, valinit=sim.charges[1])
s_q3 = Slider(ax_q3, 'Ład. B ', -10, 10, valinit=sim.charges[2])

s_G = Slider(ax_G, 'Stała ', 0.1, 10.0, valinit=1.5)
s_dt = Slider(ax_dt, 'Krok t ', 0.001, 0.05, valinit=0.01)

def update_all(val):
    sim.masses = np.array([s_m1.val, s_m2.val, s_m3.val])
    sim.charges = np.array([s_q1.val, s_q2.val, s_q3.val])
    sim.G = sim.K = s_G.val
    sim.dt = s_dt.val

for s in [s_m1, s_m2, s_m3, s_q1, s_q2, s_q3, s_G, s_dt]:
    s.on_changed(update_all)

# Wybór Trybu
rax = fig.add_axes([0.02, 0.7, 0.12, 0.08], facecolor='#f0f0f0')
radio = RadioButtons(rax, ('Grawitacja', 'Coulomb'))
def change_mode(label): sim.mode = label
radio.on_clicked(change_mode)

ax_res = fig.add_axes([0.42, 0.02, 0.16, 0.04])
btn_reset = Button(ax_res, 'ZRESETUJ UKŁAD', color="#FFFFFF", hovercolor="#000000")

def reset_sim(event):
    current_mode = radio.value_selected 
    
    sim.__init__() 
    
    sim.mode = current_mode 
    
    s_m1.reset(); s_m2.reset(); s_m3.reset()
    s_q1.reset(); s_q2.reset(); s_q3.reset()
    s_G.reset(); s_dt.reset()
    
    for i in range(3): 
        trails_p[i].set_data([], [])

btn_reset.on_clicked(reset_sim)

def animate(i):
    sim.update()
    for j in range(3):
        bodies_p[j].set_data([sim.pos[j,0]], [sim.pos[j,1]])
        bodies_p[j].set_markersize(5 + np.sqrt(sim.masses[j])*2)
        trails_p[j].set_data(sim.history[j][:,0], sim.history[j][:,1])
    return bodies_p + trails_p

ani = FuncAnimation(fig, animate, interval=15, blit=True)
plt.show()