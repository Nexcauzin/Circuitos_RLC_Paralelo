import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# A classe RLC permanece a mesma
class RLC:
    ESTIMATIVA_INICIAL = np.array([1, 1])  # Estimativa inicial para o Sistema Não Linear
    t = np.linspace(0, 10, 9000)  # Vetor do eixo de tempo

    def __init__(self, R):
        self.R = R
        self.C = 10 / 1000
        self.L = 1
        self.v_0 = 5
        self.i_0 = 0
        self.alpha = 1 / (2 * self.R * self.C)  # Frequência de neper
        self.omega = 1 / np.sqrt(self.L * self.C)  # Frequência ressonante

    # Calculando dv(0)/dt
    @property
    def dv_0(self):
        return -(self.v_0 + self.R * self.i_0) / (self.R * self.C)

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
        eq2 = self.s1 * A1 + self.s2 * A2 - self.dv_0
        return [eq1, eq2]

    def funcao_amortecimento_supercritico(self, tempo):
        """
        Função v(t) = A1*e^s1*t + A2*e^s2*t

        :param tempo: Variável de tempo
        :return:  O valor da função para t
        """
        params = self.calcular_A1_A2()
        return params[0] * np.exp(self.s1 * tempo) + params[1] * np.exp(self.s2 * tempo)

    def funcao_amortecimento_critico(self, tempo):
        """
        Função v(t) = (A1 + A2*t)*e^(-omega*t)

        :param tempo: Variável de tempo
        :return:  O valor da função para t
        """
        params = self.calcular_A1_A2()
        return (params[0] + params[1] * tempo) * np.exp(-self.omega * tempo)

    def funcao_subamortecimento(self, tempo):
        """
        Função v(t) = e^(-alpha*t)*(A1*cos(wd*t) + A2*sen(wd*t))

        :param tempo: Variável do tempo
        :return: O valor da função para t
        """

        omega_d = np.sqrt(pow(self.omega, 2) - pow(self.alpha, 2))
        params = self.calcular_A1_A2()
        return np.exp(-self.alpha * tempo) * (
            params[0] * np.cos(omega_d * tempo) + params[1] * np.sin(omega_d * tempo)
        )

    def calcular_A1_A2(self):
        if self.alpha > self.omega:
            return fsolve(self.sistema_amortecimento_supercritico, self.ESTIMATIVA_INICIAL)
        if self.alpha == self.omega:
            return [self.v_0, self.dv_0 + (self.omega * self.v_0)]
        if self.alpha < self.omega:
            omega_d = np.sqrt(pow(self.omega, 2) - pow(self.alpha, 2))
            return [self.v_0, (self.dv_0 + (self.alpha * self.v_0)) / omega_d]

    def calcula_v_t(self):
        if self.alpha > self.omega:
            return self.funcao_amortecimento_supercritico(self.t)
        if self.alpha == self.omega:
            return self.funcao_amortecimento_critico(self.t)
        if self.alpha < self.omega:
            return self.funcao_subamortecimento(self.t)

    def criar_figura_v_t(self):
        # Calcula v_t
        v_t = self.calcula_v_t()

        # Customização do gráfico
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.t, v_t, label="v(t)", color="red")
        ax.set_title("Resposta v(t)")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("v(t)")
        ax.legend()
        ax.grid(True)
        return fig

    def criar_figura_i_r(self):
        # Calcula i_r
        i_r = self.calcula_v_t() / self.R

        # Customização do gráfico
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.t, i_r, label="IR(t)", color="blue")
        ax.set_title(f"Corrente no Resistor de {self.R} Ω")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("IR(t)")
        ax.legend()
        ax.grid(True)
        return fig


# Integração com Tkinter
class RLCApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Circuitos RLC Paralelo sem Fonte")
        self.geometry("1280x720")

        self.circuitos = {
            "1.5 Ω (Supercrítico)": RLC(1.5),
            "5 Ω (Crítico)": RLC(5),
            "10 Ω (Subamortecido)": RLC(10),
            "100 Ω (Subamortecido)": RLC(100),
        }

        self.create_widgets()

    def create_widgets(self):
        # Lista para selecionar os circuitos
        self.selected_circuit = tk.StringVar(self)
        self.selected_circuit.set("1.5 Ω (Supercrítico)")

        circuit_menu = ttk.OptionMenu(
            self, self.selected_circuit, *self.circuitos.keys()
        )
        circuit_menu.pack(pady=10)

        # Botões dos gráficos
        ttk.Button(self, text="Mostrar Resposta v(t)", command=self.plot_v_t).pack(pady=5)
        ttk.Button(self, text="Mostrar Corrente IR(t)", command=self.plot_i_r).pack(pady=5)

        # Área de gráficos
        self.graph_container = tk.Frame(self)
        self.graph_container.pack(fill=tk.BOTH, expand=True)

    def plot_v_t(self):
        self._clear_graphs()
        circuito = self.circuitos[self.selected_circuit.get()]
        fig = circuito.criar_figura_v_t()
        self._display_graph(fig)

    def plot_i_r(self):
        self._clear_graphs()
        circuito = self.circuitos[self.selected_circuit.get()]
        fig = circuito.criar_figura_i_r()
        self._display_graph(fig)

    def _clear_graphs(self):
        for widget in self.graph_container.winfo_children():
            widget.destroy()

    def _display_graph(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.graph_container)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    app = RLCApp()
    app.mainloop()
