import numpy as np
import matplotlib.pyplot as plt

# — Parâmetros simplificados (ignora C1/C2) —
R2, Rb, Rout = 10e3, 100e3, 10e3
C3, Cout     = 100e-9, 100e-9

# Envelope e sinal AM
def envelope_ideal(t):
    return 1 + 1.5 * np.cos(2*np.pi*100*t)

def Vin(t):
    return envelope_ideal(t) * np.cos(2*np.pi*1000*t)

# Sistema reduzido: y = [vC3, vout]
def f(t, y):
    vC3, vout = y
    dvC3 = ((envelope_ideal(t) - vC3)/R2 - vC3/Rb) / C3
    dvout = (vC3 - vout) / (Rout * Cout)
    return np.array([dvC3, dvout])

# --- RK4 clássico ---
def rk4_step(f, t, y, h):
    k1 = f(t,          y)
    k2 = f(t + h/2,    y + h*k1/2)
    k3 = f(t + h/2,    y + h*k2/2)
    k4 = f(t + h,      y + h*k3)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# --- RK45 (Cash–Karp) ---
a = [0, 1/5, 3/10, 3/5, 1, 7/8]
b = [
    [], [1/5],
    [3/40, 9/40],
    [3/10, -9/10, 6/5],
    [-11/54, 5/2, -70/27, 35/27],
    [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]
]
c5 = [37/378, 0, 250/621, 125/594, 0, 512/1771]

def rk45_step(f, t, y, h):
    ks = []
    for i in range(6):
        ti = t + a[i]*h
        yi = y.copy()
        for j in range(i):
            yi += h * b[i][j] * ks[j]
        ks.append(f(ti, yi))
    y5 = y.copy()
    for i in range(6):
        y5 += h * c5[i] * ks[i]
    return y5

# Configuração da simulação
t0, tf, h = 0.0, 0.1, 1e-6
times = np.arange(t0, tf+h, h)
y0 = np.array([envelope_ideal(0), envelope_ideal(0)])

# Alocar soluções
sol_rk4  = np.zeros((len(times), 2))
sol_rk45 = np.zeros((len(times), 2))
sol_rk4[0]  = y0
sol_rk45[0] = y0

# Integração
for i in range(1, len(times)):
    sol_rk4[i]  = rk4_step(f, times[i-1], sol_rk4[i-1], h)
    sol_rk45[i] = rk45_step(f, times[i-1], sol_rk45[i-1], h)

# Plotagem
plt.figure()
plt.plot(times, sol_rk4[:,1],  label='Vout RK4')
plt.plot(times, sol_rk45[:,1], '--', label='Vout RK45')
plt.plot(times, envelope_ideal(times), label='Envelope Ideal', linewidth=1)
plt.xlabel('Tempo (s)')
plt.ylabel('Tensão (V)')
plt.title('Comparação: RK4 vs RK45 (h = 1e-6 s)')
plt.legend()
plt.grid(True)
plt.show()
