import numpy as np
from scipy.optimize import fsolve

# Classe RLC para gerenciar melhor os cálculos
class RLC:
    ESTIMATIVA_INICIAL = np.array([1,1])  # Estimativa inicial para o Sistema Não Linear

    def __init__(self, R):
        self.R = R
        self.C = 10/1000
        self.L = 1
        self.v_0 = 5
        self.i_0 = 0
        self.alpha = 1/(2 * self.R * self.C)  # Frequência de neper
        self.omega = 1/np.sqrt(self.L * self.C)  # Frequência ressonante

    # Calculando dv(0)/dt
    @property
    def dv_0(self):
        return -(self.v_0 + self.R * self.i_0)/(self.R * self.C)

    # Calculando as raízes s1 e s2 (Frequências Naturais)
    @property
    def s1(self):
        return -self.alpha + np.sqrt(pow(self.alpha, 2) - pow(self.omega, 2))

    @property
    def s2(self):
        return -self.alpha - np.sqrt(pow(self.alpha, 2) - pow(self.omega, 2))

    # A resposta natural é dada por:
    # v1(t) = A1 * e^s1*t
    # v2(t) = A2 * e^s2*t
    # v(t)  = v1(t) + v2(t)

    def sistema_amortecimento_supercritico(self, vars):
        """
        Resolve o sistema:
        Eq1:
        0 = A1*e^s1*t + A2*e^s2*t - v(0)
        0 = A1 + A2 - v(0)

        Eq2:
        0 = s1*A1*e^s1*t + s2*A2*e^s2*t - dv(0)/dt
        0 = s1*A1 + s2*A2 - dv(0)/dt

        :param vars: Variáveis a serem descobertas
        :return: Equações para o fsolve do scipy.optimize
        """
        (A1, A2) = vars
        eq1 = A1 + A2 - self.v_0
        eq2 = self.s1*A1 + self.s2*A2 - self.dv_0
        return [eq1, eq2]

    def calcular_A1_A2(self):
        if self.alpha > self.omega:
            return fsolve(self.sistema_amortecimento_supercritico, self.ESTIMATIVA_INICIAL)

    #def sistema_amortecimento_critico(self):


    #def sistema_subamortecido(self):


circuito_1_5_ohm = RLC(1.5) # É supercrítico
circuito_5_ohm = RLC(5) # É crítico
circuito_10_ohm = RLC(10) # Subamortecido
circuito_100_ohm = RLC(100) # Subamortecido

print(circuito_1_5_ohm.calcular_A1_A2())