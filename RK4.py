import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
C1, R1 = 0.1e-6, 100e3
Vth, rd = 0.2, 10.0    # 0.2 V limiar, 10 Ω resistência dinâmica

# Sinal de entrada
def Vin(t):
    return (1 + 1.5*np.cos(2*np.pi*20*t))*np.cos(2*np.pi*1000*t)

# Modelo piecewise-linear do diodo
def ID_pwl(Vd):
    if Vd <= Vth:
        return 0.0
    else:
        return (Vd - Vth)/rd

# Derivada dV_R/dt usando PWL
def dVdt_pwl(t, V):
    return (ID_pwl(Vin(t) - V) - V/R1)/C1

# RK4 clássico
def rk4(f, y0, t):
    y = np.zeros_like(t)
    y[0] = y0
    for i in range(len(t)-1):
        h = t[i+1] - t[i]
        k1 = f(t[i],       y[i])
        k2 = f(t[i] + h/2, y[i] + h/2*k1)
        k3 = f(t[i] + h/2, y[i] + h/2*k2)
        k4 = f(t[i] +   h, y[i] +   h*k3)
        y[i+1] = y[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y

# Simulação
t0, tf, dt = 0.0, 100e-3, 1e-6
t = np.arange(t0, tf, dt)
VR_rk4_pwl = rk4(dVdt_pwl, y0=0.0, t=t)

# Plota
plt.figure(figsize=(10,4))
plt.plot(t*1e3, VR_rk4_pwl)
plt.xlabel('Tempo (ms)')
plt.ylabel('$V_R$ (V)')
plt.title('Detector de Pico com RK4 + modelo PWL do diodo')
plt.grid(True)
plt.tight_layout()
plt.show()
