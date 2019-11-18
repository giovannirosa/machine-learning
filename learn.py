#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
import random
from datetime import datetime

import sys
import matplotlib.pyplot as plt
import seaborn as sns

from deap import base
from deap import creator
from deap import tools

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import os

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

plt.ioff()

vector_size = 672
dataframe = pd.read_csv('../credit.csv', header=0)
dataframe.drop('ID', axis=1, inplace=True)

# dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
#
# no_na_df = dataframe.fillna(-1337)
#
# no_na_df.isnull().any().any()
#
# heat_map = sns.heatmap(no_na_df)
# plt.show()
#

dataframe.describe()

dropped_df = dataframe.dropna(axis='columns').copy()

dropped_df.describe()

# for col in dropped_df.columns:
#     plt.figure(figsize=(15,8))
#     a = sns.distplot(dropped_df[col], bins =30)
#     fig = a.get_figure()
#     fig.savefig('/tmp/seaborn/{}.png'.format(col))
#     plt.close()
#     # break



# dataset = dataframe.values
dataset = dropped_df.values
# dataset = dataset.astype('float32')
X = dataset[:,1:]
np.nan_to_num(X)
# print(X[X=='NA'])
# X[X == 'NA'] = 0
y = dataset[:,0]
print(X)
# print(X[X=='NA'])
print(y)

np.any(np.isnan(X))
np.any(np.isnan(y))


train_size = int(len(dataset) * 0.5)
val_size = int(len(dataset) * 0.2)
test_size = int(len(dataset) * 0.3)

print(train_size, val_size, test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)

print(len(X_train)/len(X) * 100)
print(len(X_test)/len(X) * 100)
print(len(X_val)/len(X) * 100)


scaler = StandardScaler()
scaler.fit(X_train)





# X_train = X_train.toarray()
# X_val = X_val.toarray()

graph = []
graph2 = []


def evalOneMax(individual):

    rr = 0.0
    soma = sum(individual)

    if (soma > 0):
        mask = []
        for i in range(len(individual)):
            if(individual[i] == 0):
                mask.append(i)

        # print mask
        print("Features removed: ", len(individual) - soma)

        subX_train = np.delete(X_train, mask, axis=1)
        subX_val = np.delete(X_val, mask, axis=1)

        # clf  = Perceptron()
        clf = LinearDiscriminantAnalysis()
        clf.fit(subX_train, y_train)

        # predicao do classificador
        y_pred = clf.predict(subX_val)

        # mostra o resultado do classificador na base de teste
        rr = clf.score(subX_val, y_val)
        print("Rec Rate ",  rr)

        graph.append(1.-rr)
        graph2.append(soma)

    return rr,


def FinalTest(individual):

    print("Testing best individual on the Test Set...")
    # clf  = Perceptron()
    clf = LinearDiscriminantAnalysis()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    all_perf = clf.score(X_test, y_test)
    print("Performance Using all features ", all_perf)

    mask = []
    for i in range(len(individual)):
        if(individual[i] == 0):
            mask.append(i)

    subX_train = np.delete(X_train, mask, axis=1)
    subX_test = np.delete(X_test, mask, axis=1)

    # clf  = Perceptron()
    clf = LinearDiscriminantAnalysis()

    clf.fit(subX_train, y_train)

    # predicao do classificador
    y_pred = clf.predict(subX_test)

    # mostra o resultado do classificador na base de teste
    rr = clf.score(subX_test, y_test)
    print("Performance Using best individual ", rr)
    return (1.-rr, rr, all_perf)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, vector_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Operator registering
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


#############################################################
#############################################################
def main():

    random.seed(datetime.now())

    pop = toolbox.population(n=20)
    CXPB, MUTPB, NGEN = 0.8, 0.1, 20

    mean_perf = 0
    mean_all_perf = 0
    mean_val_perf = 0
    mean_best_ind = []
    mean_ind_qt = 0
    for i in range(0, 10):
        print("Start of evolution (%d)" % i)

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))

        np.any(np.isnan(pop))
        np.any(np.isinf(pop))

        len([y for x in pop for y in x])

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(pop))

        # Begin the evolution
        for g in range(NGEN):
            print("-- Generation %i --" % g)

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            #invalid_ind = [ind for ind in offspring]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                # print ind, fit

            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

            # graph.append(max(fits))
            # graph2.append(len(individual))

        print("-- End of (successful) evolution --")

        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, Rec Rate: %s Nb of Features: %s" %
              (best_ind, best_ind.fitness.values, sum(best_ind)))

        err, rr, all_perf = FinalTest(best_ind)

        if mean_all_perf == 0:
            mean_all_perf = all_perf
        else:
            mean_all_perf = (mean_all_perf + all_perf) / 2

        if mean_perf == 0:
            mean_perf = rr
        else:
            mean_perf = (mean_perf + rr) / 2

        if mean_val_perf == 0:
            mean_val_perf = best_ind.fitness.values[0]
        else:
            mean_val_perf = (mean_val_perf + best_ind.fitness.values[0]) / 2

        if len(mean_best_ind) == 0:
            mean_best_ind = best_ind
        else:
            for k in mean_best_ind:
                if mean_best_ind[k] != best_ind[k]:
                    mean_best_ind[k] = 1

        if mean_ind_qt == 0:
            mean_ind_qt = sum(best_ind)
        else:
            mean_ind_qt = (mean_ind_qt + sum(best_ind)) / 2
        # plt.plot(graph2, graph, 'ro')
        # plt.plot(sum(best_ind), err, 'bo')

        # plt.axis([30, vector_size-30, 0, 0.25])

        # plt.xlabel("Qtde de Caracteristicas")
        # plt.ylabel("Erro de Classificacao")

        # #line =plt.plot(graph)
        # plt.show()
    print("mean_val_perf = %f" % mean_val_perf)
    print("mean_ind_qt = %f" % mean_ind_qt)
    print("mean_val_perf - mean_perf = %f" % (mean_val_perf - mean_perf))
    print("mean_perf - mean_all_perf = %f" % (mean_perf - mean_all_perf))

    mbi_not_selected = [i + 1 for i, n in enumerate(mean_best_ind) if n==0]
    print("mean_best_ind = %s (%d)" %
          (mbi_not_selected, len(mbi_not_selected)))


if __name__ == "__main__":
    # print('Apenas um modelo, não funcional')
    main()