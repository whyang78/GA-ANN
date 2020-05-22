import numpy
import random

# Converting each solution from matrix to vector.
def mat_to_vector(mat_pop_weights):
    pop_weights_vector = []
    for sol_idx in range(mat_pop_weights.shape[0]):
        curr_vector = []
        for layer_idx in range(mat_pop_weights.shape[1]):
            vector_weights = numpy.reshape(mat_pop_weights[sol_idx, layer_idx], newshape=(mat_pop_weights[sol_idx, layer_idx].size))
            curr_vector.extend(vector_weights)
        pop_weights_vector.append(curr_vector)
    return numpy.array(pop_weights_vector)

# Converting each solution from vector to matrix.
def vector_to_mat(vector_pop_weights, mat_pop_weights):
    mat_weights = []
    for sol_idx in range(mat_pop_weights.shape[0]):
        start = 0
        end = 0
        for layer_idx in range(mat_pop_weights.shape[1]):
            end = end + mat_pop_weights[sol_idx, layer_idx].size
            curr_vector = vector_pop_weights[sol_idx, start:end]
            mat_layer_weights = numpy.reshape(curr_vector, newshape=(mat_pop_weights[sol_idx, layer_idx].shape))
            mat_weights.append(mat_layer_weights)
            start = end
    return numpy.reshape(mat_weights, newshape=mat_pop_weights.shape)

def select_mating_pool(pop, losses, num_parents):
    parents = numpy.empty((num_parents, pop.shape[1]))
    fitness_=1.0/losses
    fitness=fitness_/numpy.sum(fitness_) #通过损失计算适应度，损失越小，适应度越大
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def rws(rws_q):
    r=numpy.random.uniform(0.0,1.0)
    for i in range(rws_q.size):
        if r<rws_q[0]:
            return 0
        elif r>rws_q[i-1] and r<rws_q[i]:
            return i

def select_mating_pool_rws(pop, losses, num_parents):
    parents = numpy.empty((num_parents, pop.shape[1]))
    fitness_=1.0/losses
    fitness=fitness_/numpy.sum(fitness_) #通过损失计算适应度，损失越小，适应度越大
    rws_q=fitness.copy()

    fit_p_sum=0
    for i in range(rws_q.size):
        fit_p_sum+=rws_q[i]
        rws_q[i]=fit_p_sum

    for parent_num in range(num_parents):
        max_fitness_idx = rws(rws_q)
        parents[parent_num, :] = pop[max_fitness_idx, :]
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    #单点交叉
    crossover_point = numpy.uint32(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, mutation_percent):
    num_mutations = numpy.uint32((mutation_percent*offspring_crossover.shape[1])/100)
    mutation_indices = numpy.array(random.sample(range(0, offspring_crossover.shape[1]), num_mutations))
    for idx in range(offspring_crossover.shape[0]):
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, mutation_indices] = offspring_crossover[idx, mutation_indices] + random_value
    return offspring_crossover

#精英保留
def elitism(bestfit, bestfit_idx,bestfit_weight_vector, pop_weights_mat,pop_weights_vector, fitness):
    #存储当代适应度最小的值
    curr_bestfit = numpy.min(fitness)
    #当代适应度最小值的下标
    curr_bestfit_idx = numpy.where(fitness == numpy.min(fitness))[0]
    #当代适应度最小值的权重矩阵
    curr_bestfit_weight_mat = pop_weights_mat [curr_bestfit_idx, :]
    #当代适应度最小值与历史适应度最小值进行比较
    #如果当代适应度最小值大于历史适应度最小值
    if curr_bestfit > bestfit:
        #将当代适应度最大的权重矩阵替换为历史适应度最小的权重矩阵
        worst_idx = numpy.where(fitness == numpy.max(fitness))[0][0]
        pop_weights_vector[worst_idx, :] = bestfit_weight_vector
        bestfit_idx=worst_idx
    #如果当代适应度最小值小于历史适应度最小值
    elif curr_bestfit <= bestfit:
        #将历史适应度用当代适应度最小值替摊
        bestfit = curr_bestfit
        bestfit_idx = numpy.where(fitness == numpy.min(fitness))[0][0]
        bestfit_weight_mat = curr_bestfit_weight_mat
        bestfit_weight_vector = mat_to_vector(bestfit_weight_mat)[0]
    return pop_weights_vector, bestfit, bestfit_idx ,bestfit_weight_vector
