import numpy as np

# Parâmetros do circuito
R1, R2, Rb, Rout = 10e3, 20e3, 10e3, 10e3
Is, n, Vt, Rs = 34.9e-6, 2.28, 25.85e-3, 0.21  # BAT54
Vin_dc = 2.5  # 1 + 1.5

# Modelo de diodo e derivada
def iD(v):
    return Is * (np.exp(v/(n*Vt)) - 1) + v/Rs

def iD_prime(v):
    return Is * np.exp(v/(n*Vt)) / (n*Vt) + 1/Rs

# Função de equilíbrio F(x) = 0
def F(x):
    v1, v2, v3, v4 = x
    return np.array([
        iD(v1 - Vin_dc)     - (v1 - v4)/R1,
        iD(v1 - v2)         - (v2 - v3)/R2,
        (v2 - v3)/R2        -  v3/Rb,
        (v1 - v4)/R1        -  v4/Rout
    ])

# Jacobiana J(x)
def J(x):
    v1, v2, v3, v4 = x
    d1 = iD_prime(v1 - Vin_dc)
    d2 = iD_prime(v1 - v2)
    Jm = np.zeros((4,4))
    # F1
    Jm[0,0] =  d1 - 1/R1
    Jm[0,3] =  1/R1
    # F2
    Jm[1,0] =  d2
    Jm[1,1] = -d2 - 1/R2
    Jm[1,2] =  1/R2
    # F3
    Jm[2,1] =  1/R2
    Jm[2,2] = -1/R2 - 1/Rb
    # F4
    Jm[3,0] =  1/R1
    Jm[3,3] = -1/R1 - 1/Rout
    return Jm

# Newton–Raphson
def newton_raphson(x0, tol=1e-9, maxit=20):
    x = x0.copy()
    for k in range(maxit):
        Fx = F(x)
        if np.linalg.norm(Fx, np.inf) < tol:
            break
        Jx = J(x)
        dx = np.linalg.solve(Jx, -Fx)
        x += dx
    return x, k, Fx

# Chute inicial e execução
x0 = np.array([2.5, 2.5, 2.5, 2.5])
x_eq, iterations, residual = newton_raphson(x0)

# Jacobiana e autovalores no equilíbrio
J_eq = J(x_eq)
eigenvalues = np.linalg.eigvals(J_eq)

print("Equilíbrio:", x_eq)
print("Iterações:", iterations)
print("Resíduo:", residual)
print("Autovalores:", eigenvalues)
