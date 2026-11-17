import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

X = sp.symbols('x')
potential = 0.5 * X**2
dV = sp.diff(potential, X)
V_lambda = sp.lambdify(X, potential, 'numpy')
dV_lambda = sp.lambdify(X, dV, 'numpy')

def forward_euler(x0, v0, dt, N):
    x = [x0]
    v = [v0]
    for i in range(N):
        v_ = v[-1] - dt * dV_lambda(x[-1])
        x_ = x[-1] + dt * v[-1]
        x.append(x_)
        v.append(v_)
    return np.array(x), np.array(v)

def backward_euler(x0, v0, dt, N):
    x = [x0]
    v = [v0]
    for i in range(N):
        v_ = v[-1] - dt * dV_lambda(x[-1])
        x_ = x[-1] + dt * v_
        x.append(x_)
        v.append(v_)
    return np.array(x), np.array(v)

#  Données
x0 = 1.0
v0 = 0.0
T = 20
N = 200
dt = T / N

#  Solution exacte
def sol_exacte(t):
    return x0 * np.cos(t) + (v0) * np.sin(t)
def dsol_exacte(t):
    return -x0 * np.sin(t) + v0 * np.cos(t)

plt.figure(figsize=(10, 5))

#  Diagramme de phase
plt.subplot(1, 3, 1)
x, v = forward_euler(x0, v0, dt, N)
plt.plot(x, v)
plt.plot(sol_exacte(np.linspace(0, 2*np.pi, N+1)), dsol_exacte(np.linspace(0, 2*np.pi, N+1)), '--', c='green')
plt.legend(['Méthode d\'Euler explicite', 'Solution exacte'], loc='upper right')

#  Ajout de flèches
indices = np.linspace(0, len(x)-2, 8, dtype=int)
for i in indices:
    j = min(i + 3, len(x)-1)

    plt.annotate(
        '',
        xy=(x[j], v[j]),
        xytext=(x[i], v[i]),
        arrowprops=dict(
            arrowstyle='->',
            color='black',
            lw=1.5,
            mutation_scale=15
        )
    )

plt.xlabel('x')
plt.ylabel('v')
plt.title('Forward Euler Method')
plt.grid()

#  Energie potentielle calculée et réelle
plt.subplot(1, 3, 2)
plt.plot(x, V_lambda(x))
plt.xlabel('x')
plt.ylabel('V(x)')
plt.title('Potential Energy')
x2 = np.linspace(min(x), max(x), 100)
plt.plot(x2, V_lambda(x2))
plt.legend(['Energie potentielle calculée', 'Energie potentielle réelle'])
plt.grid()

#  Postion et énergie totale en fonction du temps
plt.subplot(1, 3, 3)
t = np.linspace(0, N*dt, N+1)
plt.plot(t, x)
plt.xlabel('Temps')
plt.ylabel('x(t)')
plt.title('Position et énergie en fonction du temps')
plt.plot(t, V_lambda(x) + 0.5 * v**2)
plt.plot(t, sol_exacte(t), '--', c='green')
plt.legend(['Position', 'Energie totale', 'Solution exacte'], loc='lower left')
plt.grid()

plt.show()



#  Consistence et convergence de la méthode d'euler explicite

fig, ax = plt.subplots(1, 2, figsize=(7, 6))
plt.subplots_adjust(bottom=0.25)

x, v = forward_euler(1, 0, dt, int(T/dt))

# Trajectoire
line, = ax[0].plot(sol_exacte(np.linspace(0, 2*np.pi, int(T/dt)+1)), dsol_exacte(np.linspace(0, 2*np.pi, int(T/dt)+1)), '--', c='green', label="Solution exacte")
line, = ax[0].plot(x, v, label="Euler explicite")
ax[0].set_xlim(-10, 10)
ax[0].set_ylim(-10, 10)
ax[0].set_aspect('equal')
ax[0].grid()
ax[0].set_xlabel('x')
ax[0].set_ylabel('v')
ax[0].set_title('Diagramme de phase')
ax[0].legend()

line2, = ax[1].plot(t, np.sqrt((x-sol_exacte(t))**2 + (v-dsol_exacte(t))**2), label="Ecart")
ax[1].set_xlabel('Temps')
ax[1].set_ylabel('Ecart')
ax[1].set_title('Ecart entre la solution exacte et la solution numérique')
ax[1].grid()

# Curseur
ax_slider = plt.axes((0.2, 0.1, 0.6, 0.03))

slider = Slider(
    ax=ax_slider,
    label='dt',
    valmin=0.001,
    valmax=1,
    valinit=dt
)

# Fonction appelée lorsque le curseur bouge
def update(val):
    dt = slider.val
    N = int(T/dt)
    t = np.arange(N+1) * dt
    x, v = forward_euler(1, 0, dt, N)

    line.set_data(x, v)
    line2.set_data(t, np.sqrt((x-sol_exacte(t))**2 + (v-dsol_exacte(t))**2))
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()


