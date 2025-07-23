import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parâmetros do circuito
C1 = 0.1e-6    # Farads (0.1 µF)
R1 = 100e3     # Ohms (100 kΩ)
IS = 2.5e-8    # A (corrente de saturação do BAT54)
n = 1.08       # fator de idealidade
VT = 25e-3     # V (tensão térmica)

# Definindo o sinal de entrada
def Vin(t):
    return (1 + 1.5 * np.cos(2 * np.pi * 20 * t)) * np.cos(2 * np.pi * 1000 * t)

# Equação diferencial dV_R/dt
def dVdt(t, V):
    Vd = Vin(t) - V
    ID = IS * (np.exp(Vd / (n * VT)) - 1)
    return (ID - V / R1) / C1

# Parâmetros da simulação
t0 = 0.0
tf = 100e-3             # simular por 100 ms
dt = 1e-6             # passo de 0.1 µs
t = np.arange(t0, tf, dt)

# Método RK4 manual
def rk4(f, y0, t):
    y = np.zeros_like(t)
    y[0] = y0
    for i in range(len(t)-1):
        h = t[i+1] - t[i]
        k1 = f(t[i],       y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2)
        k4 = f(t[i] + h,   y[i] + h * k3)
        y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return y

y0 = 0.0
VR_rk4 = rk4(dVdt, y0, t)

# RK45 com passo máximo e tolerâncias
sol = solve_ivp(
    dVdt, [t0, tf], [y0],
    method='RK45',
    t_eval=t,
    max_step=dt,
    rtol=1e-6,
    atol=1e-8
)
VR_rk45 = sol.y[0]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(t * 1e3, VR_rk45, '--', label='RK45 (max dt = 0.1 µs)')
plt.xlabel('Tempo (ms)')
plt.ylabel('$V_R$ (V)')
plt.title('Resposta do Detector de Pico usando RK45 (0–100 ms)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
