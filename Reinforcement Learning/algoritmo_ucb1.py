# O Objetivo desse código é implementar o algoritmo UCB1 (Upper Confidence Bound) através
# do problema do Multi-Armed Bandit (máquina de caça níqueis de um cassino).

from builtins import range
import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 100000
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



def ucb(mean, n, nj):                         # Essa função calcula o upper confidence bound que queremos usar para avaliar cada uma das máquinas, 
    return mean + np.sqrt(2*np.log(n) / nj)   # quando tiramos a função ARGMAX. Quanto maior o número de samples que coletamos para cada máquina,
                                              # menor fica o valor dos seus respectivos UCBs. Quanto mais próximo de 0 o UCB de uma máquina fica,
                                              # mais próximo de sua real mean o valor do cálculo (sample mean + UCB) fica. Com isso, máquinas que
                                              # foram pouco exploradas tendem a ter um UCB maior, o que nos obriga a explorá-las mais vezes. Conforme
                                              # coletamos samples para essas máquinas e o valor do UCB se aproxima de 0, mais próximo estamos de suas
                                              # real means, e mais certos estamos de qual máquina realmente é a que nos traz maior recompensa.

                                              # Detalhe: O valor de 2 é um hiperparâmetro. Quanto maior for esse valor, maior será o valor do UCB,
                                              # o que nos faz ter a necessidade de explorar mais para que cheguemos a um valor próximo de 0. Se
                                              # esse hiperparâmetro for pequeno, o nosso UCB será pequeno, o que faz com que precisemos explorar
                                              # menos para termos um UCB próximo de 0. A escolha é sua.

def experimentar():
    maquinas = [Maquina(p) for p in PROBABILIDADES_CACANIQUEIS] # Inicializando os win-rates com os valores de probabilidade.
                                                                # 3 máquinas: uma com P = 0.2, outra com P = 0.5 e outra com P = 0.75
    rewards = np.zeros(NUM_TRIALS)
    total_plays = 0

    for j in range(len(maquinas)):      # Esse loop serve para inicializar cada uma das máquinas. Para que os cálculos funcionem corretamente
        x = maquinas[j].puxar_braco()   # (para que a fórmula matemática consiga ser aplicada sem problemas), cada uma das máquinas precisa
        total_plays += 1                # ter jogado pelo menos uma vez (N precisa ser inicializado como 1).
        maquinas[j].atualizar_p(x)

    for i in range(NUM_TRIALS):
        # Selecionando a máquina com maior j, baseando-se no cálculo de seus Upper Confidence Bounds.
        j = np.argmax([ucb(m.p_estimate, total_plays, m.N) for m in maquinas])

        # Puxando o braço da máquina com a maior probabilidade
        # de vitória (exploiting). X retornado é um valor 0 ou 1,
        # para saber se vc ganhou ou não.
        x = maquinas[j].puxar_braco()

        # Somando 1 no número de vezes jogadas
        total_plays += 1
        
        # Atualizando a probabilidade para a máquina cujo
        # braço nós acabamos de puxar.
        maquinas[j].atualizar_p(x)

        # Atualizando o valor das recompensas com base na vitória ou derrota.
        rewards[i] = x

    # Cumulative Average
    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS)*np.max(PROBABILIDADES_CACANIQUEIS))
    plt.xscale('log')
    plt.xlabel('n_iter')
    plt.ylabel('mean_estimate_prob_win')
    plt.show()

    # Plotando a moving average ctr linear
    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS)*np.max(PROBABILIDADES_CACANIQUEIS))
    plt.xlabel('n_iter')
    plt.ylabel('mean_estimate_prob_win')
    plt.show()

    for m in maquinas:
        print(m.p_estimate)

    # Printando a recompensa total:
    print("Recompensa total obtida:", rewards.sum())
    print("Taxa de vitória geral:", rewards.sum() / NUM_TRIALS)
    print("Número de vezes que a máquina com o maior J foi selecionada:", [m.N for m in maquinas])

    return cumulative_average


if __name__ == "__main__":
    experimentar()
