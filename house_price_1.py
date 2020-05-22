import numpy
import GA
import ANN
import pickle
import matplotlib.pyplot
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
numpy.random.seed(78)

#加载数据 波斯顿房价数据 并划分训练集与测试集
data = load_boston()
X = data.data
y = data.target
ss = StandardScaler()
X = ss.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=78)

"""
Genetic algorithm parameters:
    Mating Pool Size (Number of Parents)
    Population Size
    Number of Generations
    Mutation Percent
"""
sol_per_pop = 8 # 种群数目
num_parents_mating = 4 # 父母数目
num_generations = 2000 # 遗传代数
mutation_percent = 5 # 变异比例

#创建种群 将网络权重转换成向量
initial_pop_weights = []
for curr_sol in numpy.arange(0, sol_per_pop):
    HL1_neurons = 100
    input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1,
                                             size=(X_train.shape[1], HL1_neurons))
    HL2_neurons = 50
    HL1_HL2_weights = numpy.random.uniform(low=-0.1, high=0.1,
                                             size=(HL1_neurons, HL2_neurons))

    output_neurons = 1
    HL2_output_weights = numpy.random.uniform(low=-0.1, high=0.1,
                                              size=(HL2_neurons, output_neurons))

    initial_pop_weights.append(numpy.array([input_HL1_weights,
                                                HL1_HL2_weights,
                                                HL2_output_weights]))

pop_weights_mat = numpy.array(initial_pop_weights)
pop_weights_vector = GA.mat_to_vector(pop_weights_mat)

# 遗传算法优化网络权重
bestfit=999999999
bestfit_idx=0
bestfit_weight_vector=None
losses= numpy.empty(shape=(num_generations))
for generation in range(num_generations-1):
    print("Generation : ", generation)

    # 网络权重
    pop_weights_mat = GA.vector_to_mat(pop_weights_vector,
                                       pop_weights_mat)

    # 得到每一个种群的损失 损失值是适应度 则适应度越小越好
    loss_value = ANN.fitness(pop_weights_mat,
                          X_train,
                          y_train,
                          activation="relu")

    # 精英策略
    pop_weights_vector, bestfit, bestfit_idx,bestfit_weight_vector=GA.elitism(bestfit,bestfit_idx, bestfit_weight_vector, pop_weights_mat, pop_weights_vector, loss_value)
    losses[generation] = bestfit

    # 选择最佳父母
    parents = GA.select_mating_pool_rws(pop_weights_vector,
                                    loss_value.copy(),
                                    num_parents_mating)

    # 通过交叉生成后代
    offspring_crossover = GA.crossover(parents,
                                       offspring_size=(pop_weights_vector.shape[0]-parents.shape[0], pop_weights_vector.shape[1]))

    # 变异处理
    offspring_mutation = GA.mutation(offspring_crossover,
                                     mutation_percent=mutation_percent)

    # 通过父母和后代生成新的种群
    pop_weights_vector[0:parents.shape[0], :] = parents
    pop_weights_vector[parents.shape[0]:, :] = offspring_mutation


pop_weights_vector, bestfit, bestfit_idx,bestfit_weight_vector=GA.elitism(bestfit,bestfit_idx, bestfit_weight_vector, pop_weights_mat, pop_weights_vector, loss_value)
losses[num_generations-1] = bestfit
# 将向量转成成权重矩阵
pop_weights_mat = GA.vector_to_mat(pop_weights_vector, pop_weights_mat)
best_weights = pop_weights_mat [bestfit_idx, :]
# 得到训练损失与训练预测结果
train_loss,train_pred= ANN.predict_outputs(best_weights, X_train, y_train, activation="relu")
print("train loss : ", train_loss)
#绘制真实结果与预测结果对比图
matplotlib.pyplot.figure(1)
matplotlib.pyplot.scatter(y_train, train_pred)
x = numpy.linspace(0,60,10)
matplotlib.pyplot.plot(x,x,color='red',linestyle='--',linewidth=2.5)
matplotlib.pyplot.xlim(0,60)
matplotlib.pyplot.ylim(0,60)
matplotlib.pyplot.xlabel("real", fontsize=20)
matplotlib.pyplot.ylabel("predict", fontsize=20)
matplotlib.pyplot.show()

# 得到测试损失与测试预测结果
test_loss,test_pred = ANN.predict_outputs(best_weights, X_test, y_test, activation="relu")
print("test loss : ", test_loss)
matplotlib.pyplot.figure(2)
matplotlib.pyplot.scatter(y_test, test_pred)
x = numpy.linspace(0,60,10)
matplotlib.pyplot.plot(x,x,color='red',linestyle='--',linewidth=2.5)
matplotlib.pyplot.xlim(0,60)
matplotlib.pyplot.ylim(0,60)
matplotlib.pyplot.xlabel("real", fontsize=20)
matplotlib.pyplot.ylabel("predict", fontsize=20)
matplotlib.pyplot.show()

# 绘制遗传不同代数对应的损失
matplotlib.pyplot.figure(3)
matplotlib.pyplot.plot(losses, linewidth=5, color="black")
matplotlib.pyplot.xlabel("Iteration", fontsize=20)
matplotlib.pyplot.ylabel("losses", fontsize=20)
# matplotlib.pyplot.xticks(numpy.arange(0, num_generations+1, 100), fontsize=15)
matplotlib.pyplot.show()

# 保存权重
f = open("weights_"+str(num_generations)+"_iterations_"+str(mutation_percent)+"%_mutation.pkl", "wb")
pickle.dump(pop_weights_mat, f)
f.close()
