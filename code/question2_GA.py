import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# 创建适应度函数的最大化目标
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# 创建个体（染色体）类
creator.create("Individual", list, fitness=creator.FitnessMax)

# 定义问题参数
num_genes = 6  # 决策变量的数量


# 定义目标函数（适应度函数）
def objective_function(individual):
    # 对应决策变量
    x = individual

    c1 = 0.004  # 常数c1
    c2 = [12.3533066, 3.225434281, 7.925177134, 4.58402071, 3.614925942, 3.724212133]  # c2数组
    a = [25.452, 279.4170, 69.2915, 9.3025, 59.0226, 106.723]  # a
    b = [0.0421, -0.0778, -0.0639, 0.0898, 0.0395, -0.0445]  # b
    loss = [0.11975, 0.1028, 0.14142, 0.07122, 0.08515, 0.08131]  # loss数组
    alpha = [0.276178142, 0.614610374, 0.499660436, 0.681622204, 0.715190645, 0.46527093]  # alpha数组

    total_value = 0
    for i in range(num_genes):
        total_value += (((c1 / x[i] + c2[i]) * (1 + alpha[i]) - c2[i]) * a[i] * np.exp(b[i] * ((c1 / x[i] + c2[i]) * (1 + alpha[i]))) - loss[i] * x[i] *
                       c2[i])

    return total_value,


# 定义约束条件函数
def constraint_1(individual):
    # 对应决策变量
    x = individual

    qmax = [18.90426096, 146.0112251, 26.23555554, 24.50013194, 108.932873, 53.35858281]
    qmin = [15.4671226, 119.4637296, 21.46545453, 20.0455625, 89.12689608, 43.6570223]

    for i in range(num_genes):
        if x[i] < qmin[i] or x[i] > qmax[i]:
            return False
    return True


# 定义遗传算法的进化过程
def genetic_algorithm(pop_size, num_generations):
    toolbox = base.Toolbox()

    # 注册每个决策变量的范围
    qmax = [18.90426096, 146.0112251, 26.23555554, 24.50013194, 108.932873, 53.35858281]
    qmin = [15.4671226, 119.4637296, 21.46545453, 20.0455625, 89.12689608, 43.6570223]

    for i in range(num_genes):
        gene_name = f"gene{i + 1}"
        toolbox.register(gene_name, random.uniform, qmin[i], qmax[i])

    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.gene1, toolbox.gene2, toolbox.gene3, toolbox.gene4, toolbox.gene5, toolbox.gene6), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective_function)
    toolbox.decorate("evaluate", tools.DeltaPenalty(constraint_1, (0.0,)))
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[1] * num_genes, up=[10] * num_genes, eta=1.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=[1] * num_genes, up=[10] * num_genes, eta=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)

    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_generations, verbose=False)
    # 初始化迭代收敛曲线数据
    best_fitness_values = []

    for gen in range(num_generations):
        # 迭代进化
        algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=1, verbose=False)

        # 记录每一代的最佳适应度值
        best_individual = tools.selBest(population, k=1)[0]
        best_fitness = best_individual.fitness.values[0]
        best_fitness_values.append(best_fitness)

        print(f"代数 {gen + 1}/{num_generations} - 最佳适应度值: {best_fitness}")

    # 绘制迭代收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_generations + 1), best_fitness_values, marker='o', linestyle='-')
    plt.title("迭代收敛曲线")
    plt.xlabel("代数")
    plt.ylabel("最佳适应度值")
    plt.grid(True)
    plt.show()
    best_individual = tools.selBest(population, k=1)[0]
    best_fitness = best_individual.fitness.values[0]

    return best_individual, best_fitness


if __name__ == "__main__":
    # 设置随机数生成器的种子，例如，使用种子值11
    random.seed(11)

    pop_size = 50  # 种群大小
    num_generations = 500  # 进化的代数

    best_solution, best_fitness = genetic_algorithm(pop_size, num_generations)

    print("最佳解决方案:", best_solution)
    print("最佳适应度值:", best_fitness)


