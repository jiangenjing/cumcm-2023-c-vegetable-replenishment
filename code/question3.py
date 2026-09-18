import pandas as pd
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt


# 读取进价表格.csv文件
df = pd.read_csv('6.24-6.30单品销售量.csv',encoding='gbk')

# 获取单品名称列（除去第一行）
item_names = df.columns[1:]

# 定义GM(1,1)灰色预测模型
def GM_11(x0):
    n = len(x0)
    x1 = np.cumsum(x0)
    Z1 = np.zeros((n-1, 2))
    for i in range(1, n):
        Z1[i-1][0] = -0.5 * (x1[i] + x1[i-1])
        Z1[i-1][1] = 1
    B = np.linalg.inv(Z1.T @ Z1) @ Z1.T @ x0[1:]
    a, u = B[0], B[1]
    X_ = np.zeros(n)
    X_[0] = x0[0]
    for i in range(1, n):
        X_[i] = (x0[0] - u/a) * (1 - np.exp(a)) * np.exp(-a*(i))
    return X_

# 预测未来1天的值
predicted_values = {}
for item_name in item_names:
    x0 = df[item_name].values
    predicted_value = GM_11(x0)[-1]
    predicted_values[item_name] = predicted_value

# 创建包含预测结果的DataFrame
prediction_df = pd.DataFrame.from_dict(predicted_values, orient='index', columns=['预测值'])

# 导出结果为CSV文件
prediction_df.to_csv('销售量灰色预测.csv', index=True)



plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 创建适应度函数的最大化目标
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# 创建个体（染色体）类
creator.create("Individual", list, fitness=creator.FitnessMax)

# 定义问题参数
num_genes = 49  # 决策变量的数量

# 定义目标函数（适应度函数）
def objective_function(individual):
    # 对应决策变量
    x = individual

    c1 = 0.004  # 常数c1
    c2 = [
        2.11, 7.59, 15.6, 12.13, 1.3, 7, 8.39, 3.43, 8.97, 2.84,
        3.49, 3.63, 4.09, 10.38, 2.33, 2.56, 2.15, 4.29, 9.16, 3.41,
        4.07, 2.6, 3.12, 4.4, 2.7, 2.1, 18, 3.32, 6.68, 2.21, 11.67,
        1.45, 1.95, 3.29, 4.62, 12.72, 3.12, 2.6, 1.53, 3.2, 2.55, 3.98,
        2.66, 4.34, 5.29, 9.66, 9.9, 13.69, 16.06
    ]  # c2数组
    a = 106.723  # a
    b = -0.0445  # b
    loss = [
        0.0943, 0.0926, 0.108, 0.0943, 0.0943, 0.069, 0.0943, 0.0607, 0.1018, 0.0943,
        0.0943, 0.057, 0.1443, 0.0554, 0.0943, 0.1568, 0.1362, 0.0943, 0.0961, 0.002,
        0.0943, 0.0842, 0.0761, 0.0248, 0.1033, 0.0943, 0.2405, 0.0943, 0.1525, 0.1852,
        0.2925, 0.0045, 0, 0.0657, 0.137, 0.0943, 0.0084, 0.0943, 0.0943, 0.0671, 0.0943,
        0.0501, 0.1281, 0.0943, 0.1663, 0.1851, 0.2616, 0.0943, 0.1269
    ]  # loss数组
    alpha = [
        1.720412579, 0.63297572, 0.538461538, 0.598011048, 3.846153846, 0.716734694,
        0.466848118, 0.656809663, 0.436215958, 0.450001818, 0.239266345, 0.500196773,
        0.948305973, 0.347481071, 0.959600762, 0.887369792, 0.662458472, 0.393789669,
        0.526928675, 0.470081951, 0.307197379, 0.833516484, 0.729190505, 0.406606223,
        0.877777778, 0.507518457, 0.14934498, 0.726549053, 0.518712575, 0.641887524,
        0.170697596, 0.280581169, 0.390316497, 1.132218845, 0.301406926, 0.46079655,
        0.65357906, 0.440769231, 1.549019608, 0.7984375, 0.576470588, 0.490368509,
        1.250626566, 1.073732719, 0.680529301, 0.451345756, 0.424915825, 0.239277656,
        0.618721461
    ]  # alpha数组

    total_value = 0
    for i in range(num_genes):
        total_value += 0.1*((((c1 / x[i] + c2[i]) * (1 + alpha[i]) - c2[i]) * a * np.exp(b * ((c1 / x[i] + c2[i]) * (1 + alpha[i]))) - loss[i] * x[i] * \
                       c2[i]))

    return total_value,

# 定义约束条件函数
def constraint_1(individual):
    # 对应决策变量
    x = individual

    qmax = [
        25.3, 18.59, 7.2292, 7.7396, 10.67, 9.6679, 12.3057, 15.8015, 9.317, 27.5,
        39.6, 17.4856, 7.7396, 7.084, 9.9, 9.9484, 14.1724, 11, 3.7356, 9.9,
        12.1, 6.468, 5.7035, 6.6, 5.1799, 11, 4.7729, 4.4, 3.1944, 7.3986,
        4.158, 14.3, 6.6, 1.1, 2.9711, 2.112, 2.2, 2.2, 1.1, 1.5136, 1.1,
        1.034, 0.3223, 0.5632, 0.7931, 1.4025, 1.0956, 0.5632, 0.2541
    ]
    qmin = [
        20.7, 15.21, 5.9148, 6.3324, 8.73, 7.9101, 10.0683, 12.9285, 7.623, 22.5,
        32.4, 14.3064, 6.3324, 5.796, 8.1, 8.1396, 11.5956, 9, 3.0564, 8.1,
        9.9, 5.292, 4.6665, 5.4, 4.2381, 9, 3.9051, 3.6, 2.6136, 6.0534,
        3.402, 11.7, 5.4, 0.9, 2.4309, 1.728, 1.8, 1.8, 0.9, 1.2384, 0.9,
        0.846, 0.2637, 0.4608, 0.6489, 1.1475, 0.8964, 0.4608, 0.2079
    ]

    for i in range(num_genes):
        if x[i] < qmin[i] or x[i] > qmax[i]:
            return False
    return True

# 定义遗传算法的进化过程
def genetic_algorithm(pop_size, num_generations):
    toolbox = base.Toolbox()

    qmax = [
        25.3, 18.59, 7.2292, 7.7396, 10.67, 9.6679, 12.3057, 15.8015, 9.317, 27.5,
        39.6, 17.4856, 7.7396, 7.084, 9.9, 9.9484, 14.1724, 11, 3.7356, 9.9,
        12.1, 6.468, 5.7035, 6.6, 5.1799, 11, 4.7729, 4.4, 3.1944, 7.3986,
        4.158, 14.3, 6.6, 1.1, 2.9711, 2.112, 2.2, 2.2, 1.1, 1.5136, 1.1,
        1.034, 0.3223, 0.5632, 0.7931, 1.4025, 1.0956, 0.5632, 0.2541
    ]
    qmin = [
        20.7, 15.21, 5.9148, 6.3324, 8.73, 7.9101, 10.0683, 12.9285, 7.623, 22.5,
        32.4, 14.3064, 6.3324, 5.796, 8.1, 8.1396, 11.5956, 9, 3.0564, 8.1,
        9.9, 5.292, 4.6665, 5.4, 4.2381, 9, 3.9051, 3.6, 2.6136, 6.0534,
        3.402, 11.7, 5.4, 0.9, 2.4309, 1.728, 1.8, 1.8, 0.9, 1.2384, 0.9,
        0.846, 0.2637, 0.4608, 0.6489, 1.1475, 0.8964, 0.4608, 0.2079
    ]

    for i in range(num_genes):
        gene_name = f"gene{i + 1}"
        toolbox.register(gene_name, random.uniform, qmin[i], qmax[i])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.gene1, toolbox.gene2, toolbox.gene3, toolbox.gene4, toolbox.gene5,
                      toolbox.gene6, toolbox.gene7, toolbox.gene8, toolbox.gene9, toolbox.gene10,
                      toolbox.gene11, toolbox.gene12, toolbox.gene13, toolbox.gene14, toolbox.gene15,
                      toolbox.gene16, toolbox.gene17, toolbox.gene18, toolbox.gene19, toolbox.gene20,
                      toolbox.gene21, toolbox.gene22, toolbox.gene23, toolbox.gene24, toolbox.gene25,
                      toolbox.gene26, toolbox.gene27, toolbox.gene28, toolbox.gene29, toolbox.gene30,
                      toolbox.gene31, toolbox.gene32, toolbox.gene33, toolbox.gene34, toolbox.gene35,
                      toolbox.gene36, toolbox.gene37, toolbox.gene38, toolbox.gene39, toolbox.gene40,
                      toolbox.gene41, toolbox.gene42, toolbox.gene43, toolbox.gene44, toolbox.gene45,
                      toolbox.gene46, toolbox.gene47, toolbox.gene48, toolbox.gene49), n=1)

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
        best_fitness = best_individual.fitness.values[0]-203.93
        best_fitness_values.append(best_fitness)

        print(f"代数 {gen + 1}/{num_generations} - 最佳适应度值: {best_fitness}")

    # 绘制迭代收敛曲线
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, num_generations + 1), best_fitness_values, marker='o', linestyle='-')
    plt.title("迭代收敛曲线",fontsize=16)
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

