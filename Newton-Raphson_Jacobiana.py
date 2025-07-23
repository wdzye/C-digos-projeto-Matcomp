import numpy as np

# Parâmetros do circuito
R1 = 100e3    # 100 kΩ
C1 = 0.1e-6    # 0.1 μF
I_S = 1e-9     # Corrente de saturação do diodo BAT54
n = 1.0        # Fator de idealidade
V_T = 0.02585  # Tensão térmica a 25°C (298 K)

def equilibrium_voltage(V_in, V_R0=0.0, tol=1e-6, max_iter=100):
    """
    Calcula a tensão de equilíbrio VR usando Newton-Raphson.
    """
    V_R = V_R0
    for _ in range(max_iter):
        exp_term = np.exp((V_in - V_R) / (n * V_T))
        f = I_S * (exp_term - 1) - V_R / R1
        df = - (I_S / (n * V_T)) * exp_term - 1 / R1
        delta = f / df
        V_R -= delta
        if abs(delta) < tol:
            break
    return V_R

def jacobian_eigenvalue(V_in):
    """
    Calcula o autovalor da Jacobiana no ponto de equilíbrio.
    
    Retorna:
        autovalor: Valor do autovalor (escalar)
        V_R_eq: Tensão de equilíbrio
        stability: String indicando a estabilidade
    """
    # Passo 1: Encontrar o ponto de equilíbrio
    V_R_eq = equilibrium_voltage(V_in)
    
    # Passo 2: Calcular a Jacobiana no ponto de equilíbrio
    exp_term = np.exp((V_in - V_R_eq) / (n * V_T))
    dF_dVR = - (I_S / (n * V_T)) * exp_term - 1 / R1
    jacobian = dF_dVR / C1  # Autovalor da linearização
    
    # Determinar estabilidade
    stability = "estável" if jacobian < 0 else "instável"
    
    return jacobian, V_R_eq, stability

# Exemplo de uso:
if __name__ == "__main__":
    V_in_test = 2.0  # Tensão de entrada para teste
    
    # Calcular autovalor e estabilidade
    eigenvalue, V_R_eq, stability = jacobian_eigenvalue(V_in_test)
    
    print("\n" + "="*50)
    print("Análise de Estabilidade do Ponto de Equilíbrio")
    print("="*50)
    print(f"Tensão de entrada (V_in): {V_in_test} V")
    print(f"Ponto de equilíbrio (V_R): {V_R_eq:.6f} V")
    print(f"Autovalor da Jacobiana: {eigenvalue:.4f}")
    print(f"Conclusão: O ponto de equilíbrio é {stability} (autovalor {'<' if eigenvalue < 0 else '>'} 0)")
    print("="*50)
