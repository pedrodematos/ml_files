# O Objetivo desse código é implementar o algoritmo Epsilon-Greedy através
# do problema do Multi-Armed Bandit (máquina de caça níqueis de um cassino).

from builtins import range
import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 10000
EPSILON = 0.1
PROBABILIDADES_CACANIQUEIS = [0.2, 0.5, 0.75]


class Maquina:
    def __init__(self, p):
        self.p = p # P significa o Win-Rate da máquina
        self.p_estimate = 0. # P atualizado após os jogos
        self.N = 0. # Número de samples coletadas para cada máquina até agora

    def puxar_braco(self):
        # Retorna um booleano (entre 0 ou 1) com probabilidade P (pra ver se vc ganhou ou não.
        # Se o número aleatório for menor que a probabilidade P, você ganha. Se for maior, perde.)
        # é uma forma resumida de escrever o código abaixo:
        # n_aleatorio = np.random.random()  -> retorna um número aleatório entre 0 e 1
        # if (n_aleatorio < self.p):
        #    return 1
        # else:
        #    return 0    
        return np.random.random() < self.p

    def atualizar_p(self, x): # Essa função atualiza o winrate da máquina. O valor de X pode ser 0 ou 1 (ganhou ou não).
        self.N += 1 # Somando 1 ao número de samples para a determinada máquina
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N # Atualizando o valor de p e atribuindo ao p_estimate, utilizando
                                                                        # a fórmula da Sample Mean recursiva (que toma como base o valor da última
                                                                        # mean e o total de registros coletados até o momento para calcular a atual).

def experimentar():
    maquinas = [Maquina(p) for p in PROBABILIDADES_CACANIQUEIS] # Inicializando os win-rates com os valores de probabilidade.
                                                                # 3 máquinas: uma com P = 0.2, outra com P = 0.5 e outra com P = 0.75

    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0

    optimal_j = np.argmax([m.p for m in maquinas]) # Valor correspondente ao da máquina com o maior win-rate
    print("Valor ótimo de J, máquina do índice:", optimal_j)

    for i in range(NUM_TRIALS):

        # Se eu gerar um número aleatório menor
        # que o meu epsilon, selecione aleatoriamente
        # uma máquina para se jogar.
        if np.random.random() < EPSILON:
            num_times_explored += 1 # Contando o número de explorações
            j = np.random.randint(len(maquinas)) # Selecionando uma das 3 máquinas aleatoriamente como j

        else:
            num_times_exploited += 1 # Contando o número de exploitations (o valor aleatório gerado acima foi maior ou igual a 0.1)
            j = np.argmax([m.p_estimate for m in maquinas])

        if j == optimal_j:
            num_optimal += 1 # Número de vezes que, mesmo com o Epsilon-Greedy, a máquina com maior J foi selecionada

        # Puxando o braço da máquina com a maior probabilidade
        # de vitória (exploiting). X retornado é um valor 0 ou 1,
        # para saber se vc ganhou ou não.
        x = maquinas[j].puxar_braco()

        # Atualizando o valor das recompensas com base na vitória ou derrota.
        rewards[i] = x
        
        # Atualizando a probabilidade para a máquina cujo
        # braço nós acabamos de puxar.
        maquinas[j].atualizar_p(x)


    # Printando a média estimada para cada máquina:
    for m in maquinas:
        print("Estimativa média:", m.p_estimate)

    # Printando a recompensa total:
    print("Recompensa total obtida:", rewards.sum())
    print("Taxa de vitória geral:", rewards.sum() / NUM_TRIALS)
    print("Número de explorations:", num_times_explored)
    print("Número de exploitations:", num_times_exploited)
    print("Número de vezes que a máquina com o maior J foi selecionada:", num_optimal)

    # Plotando os resultados:
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) + np.max(PROBABILIDADES_CACANIQUEIS))
    plt.xlabel('n_iter')
    plt.ylabel('mean_estimate_prob_win')
    plt.show()


if __name__ == "__main__":
    experimentar()
