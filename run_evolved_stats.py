import sys
import numpy as np
import scipy.stats
import operator
import itertools
import math

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

np.seterr(divide='ignore')

#def runGA():
# Generate the distributions to be used in the GP optimization process
# - Normal distribution
# - Equal variance
# - Differing means

# Sample distribution for test power metric
# Index by: [sample number (0-29)]
sig_diff_samples_0v1_same_std = np.random.normal(loc=20, scale=1, size=(30, 100))
sig_diff_samples_1v0_same_std = np.random.normal(loc=21, scale=1, size=(30, 100))

sig_diff_samples_0v2_same_std = np.random.normal(loc=30, scale=2, size=(30, 100))
sig_diff_samples_2v0_same_std = np.random.normal(loc=32, scale=2, size=(30, 100))

sig_diff_samples_0v4_same_std = np.random.normal(loc=40, scale=4, size=(30, 100))
sig_diff_samples_4v0_same_std = np.random.normal(loc=44, scale=4, size=(30, 100))

# Sample distributions for scale invariance metric
sig_diff_samples_0v10_same_std = sig_diff_samples_0v1_same_std * 10.
sig_diff_samples_10v0_same_std = sig_diff_samples_1v0_same_std * 10.

sig_diff_samples_0v100_same_std = sig_diff_samples_0v1_same_std * 100.
sig_diff_samples_100v0_same_std = sig_diff_samples_1v0_same_std * 100.

# Index by: [group (0/1)][sample number (0-29)]
null_samples_0v1_same_std = [[], []]
for dist1, dist2 in zip(sig_diff_samples_0v1_same_std, sig_diff_samples_1v0_same_std):
    both_dist = np.copy(np.append(dist1, dist2))
    np.random.shuffle(both_dist)
    dist1_sample = both_dist[:int(len(both_dist) / 2.)]
    dist2_sample = both_dist[int(len(both_dist) / 2.):]
    null_samples_0v1_same_std[0].append(dist1_sample)
    null_samples_0v1_same_std[1].append(dist2_sample)
null_samples_0v1_same_std = np.array(null_samples_0v1_same_std)

null_samples_0v2_same_std = [[], []]
for dist1, dist2 in zip(sig_diff_samples_0v2_same_std, sig_diff_samples_2v0_same_std):
    both_dist = np.copy(np.append(dist1, dist2))
    np.random.shuffle(both_dist)
    dist1_sample = both_dist[:int(len(both_dist) / 2.)]
    dist2_sample = both_dist[int(len(both_dist) / 2.):]
    null_samples_0v2_same_std[0].append(dist1_sample)
    null_samples_0v2_same_std[1].append(dist2_sample)
null_samples_0v2_same_std = np.array(null_samples_0v2_same_std)

null_samples_0v4_same_std = [[], []]
for dist1, dist2 in zip(sig_diff_samples_0v4_same_std, sig_diff_samples_4v0_same_std):
    both_dist = np.copy(np.append(dist1, dist2))
    np.random.shuffle(both_dist)
    dist1_sample = both_dist[:int(len(both_dist) / 2.)]
    dist2_sample = both_dist[int(len(both_dist) / 2.):]
    null_samples_0v4_same_std[0].append(dist1_sample)
    null_samples_0v4_same_std[1].append(dist2_sample)
null_samples_0v4_same_std = np.array(null_samples_0v4_same_std)

# GP tree: takes two arrays as input, returns a test staistic
pset = gp.PrimitiveSetTyped('MAIN', [np.ndarray, np.ndarray], float)
pset.renameArguments(ARG0='x1')
pset.renameArguments(ARG1='x2')

# Logical operators on the distance array
#pset.addPrimitive(np.logical_and, [np.ndarray, np.ndarray], np.ndarray, name='array_and')
#pset.addPrimitive(np.logical_or, [np.ndarray, np.ndarray], np.ndarray, name='array_or')
#pset.addPrimitive(np.logical_xor, [np.ndarray, np.ndarray], np.ndarray, name='array_xor')
#pset.addPrimitive(np.logical_not, [np.ndarray], np.ndarray, name='array_not')

# Mathematical operators on the distance array
#pset.addPrimitive(np.add, [np.ndarray, np.ndarray], np.ndarray, name='array_add')
#pset.addPrimitive(np.subtract, [np.ndarray, np.ndarray], np.ndarray, name='array_sub')
#pset.addPrimitive(np.multiply, [np.ndarray, np.ndarray], np.ndarray, name='array_mul')
#pset.addPrimitive(np.divide, [np.ndarray, np.ndarray], np.ndarray, name='array_div')
pset.addPrimitive(np.sqrt, [np.ndarray], np.ndarray, name='array_sqrt')
pset.addPrimitive(np.square, [np.ndarray], np.ndarray, name='array_square')
pset.addPrimitive(np.abs, [np.ndarray], np.ndarray, name='array_abs')

# Statistics derived from the distance array
pset.addPrimitive(np.mean, [np.ndarray], float, name='array_mean')
pset.addPrimitive(np.median, [np.ndarray], float, name='array_median')
pset.addPrimitive(np.min, [np.ndarray], float, name='array_min')
pset.addPrimitive(np.max, [np.ndarray], float, name='array_max')
pset.addPrimitive(np.std, [np.ndarray], float, name='array_std')
pset.addPrimitive(np.var, [np.ndarray], float, name='array_var')
pset.addPrimitive(np.size, [np.ndarray], float, name='array_size')
pset.addPrimitive(np.sum, [np.ndarray], float, name='array_sum')
pset.addPrimitive(scipy.stats.sem, [np.ndarray], float, name='array_stderr')

# Mathematical operators with single values
def protected_div(left, right):
    try:
        return float(left) / float(right)
    except ZeroDivisionError:
        return 1.

pset.addPrimitive(operator.add, [float, float], float, name='float_add')
pset.addPrimitive(operator.sub, [float, float], float, name='float_sub')
pset.addPrimitive(operator.mul, [float, float], float, name='float_mul')
pset.addPrimitive(protected_div, [float, float], float, name='float_div')
pset.addPrimitive(np.sqrt, [float], float, name='float_sqrt')
pset.addPrimitive(np.square, [float], float, name='float_square')
pset.addPrimitive(np.abs, [float], float, name='float_abs')

# Mathematical operators on the distance array with a single value
pset.addPrimitive(np.add, [np.ndarray, float], np.ndarray, name='array_add_float')
pset.addPrimitive(np.subtract, [np.ndarray, float], np.ndarray, name='array_sub_float')
pset.addPrimitive(np.multiply, [np.ndarray, float], np.ndarray, name='array_mul_float')
pset.addPrimitive(np.divide, [np.ndarray, float], np.ndarray, name='array_div_float')

# Equivalence operators on the distance array with a single value
#pset.addPrimitive(np.less, [np.ndarray, float], np.ndarray, name='array_less_than_float')
#pset.addPrimitive(np.equal, [np.ndarray, float], np.ndarray, name='array_equal_float')

# Terminals
pset.addTerminal(1.0, float)
#pset.addEphemeralConstant('rand{}'.format(np.random.randint(1e9)), lambda: np.random.random() * 100., float)
#pset.addTerminal(np.multiply(np.random.random(size=features.shape[0]), 100.), np.ndarray)
#pset.addTerminal(np.array([True] * features.shape[0]), np.ndarray)
#pset.addTerminal(np.array([False] * features.shape[0]), np.ndarray)

creator.create('FitnessMulti', base.Fitness, weights=(1., -1., -1.))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=1, max_=4)

"""
def return_ttest():
    return creator.Individual.from_string('float_div(float_sub(array_mean(x1), array_mean(x2)), float_add(array_stderr(x1), array_stderr(x2)))', pset)
    #return creator.Individual.from_string('float_div(float_sub(array_mean(x1), array_mean(x2)), float_sqrt(float_add(float_div(array_var(x1), array_size(x1)), float_div(array_var(x2), array_size(x2)))))', pset)

toolbox.register('individual', return_ttest)
"""
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gp.compile, pset=pset)

total_skipped = 0
total_cache_lookups = 0
solution_cache = {}

def evaluate_individual(individual):
    global total_skipped
    global total_cache_lookups
    global solution_cache
    
    #if (str_ind == 'float_div(float_sub(array_mean(x1), array_mean(x2)), float_sqrt(float_add(float_div(array_var(x1), array_size(x1)), float_div(array_var(x2), array_size(x2)))))' or
    #    str_ind == 'float_div(float_sub(array_mean(x1), array_mean(x2)), float_add(array_stderr(x1), array_stderr(x2)))'):
    #    print('t-test!')
    
    if str(individual) in solution_cache:
        total_cache_lookups += 1
        return solution_cache[str(individual)]

    func = toolbox.compile(expr=individual)

    # First fitness component:
    #     - Test statistic for sig diff means should be well outside the null distribution
    #     - Test statistic for non sig diff means should be in the middle of the null distribution
    ts_0v1_same_std = []
    ts_0v2_same_std = []
    ts_0v4_same_std = []
    ts_0v10_same_std = []
    ts_0v100_same_std = []
    ts_null_0v1_same_std = []
    ts_null_0v2_same_std = []
    ts_null_0v4_same_std = []

    # Sig diff sample comparisons
    for sample1, sample2 in zip(sig_diff_samples_0v1_same_std, sig_diff_samples_1v0_same_std):
        ts_0v1_same_std.append(func(sample1, sample2))
        ts_0v1_same_std.append(func(sample2, sample1))

    for sample1, sample2 in zip(sig_diff_samples_0v2_same_std, sig_diff_samples_2v0_same_std):
        ts_0v2_same_std.append(func(sample1, sample2))
        ts_0v2_same_std.append(func(sample2, sample1))

    for sample1, sample2 in zip(sig_diff_samples_0v4_same_std, sig_diff_samples_4v0_same_std):
        ts_0v4_same_std.append(func(sample1, sample2))
        ts_0v4_same_std.append(func(sample2, sample1))
    
    for sample1, sample2 in zip(sig_diff_samples_0v10_same_std, sig_diff_samples_10v0_same_std):
        ts_0v10_same_std.append(func(sample1, sample2))
        ts_0v10_same_std.append(func(sample2, sample1))

    for sample1, sample2 in zip(sig_diff_samples_0v100_same_std, sig_diff_samples_100v0_same_std):
        ts_0v100_same_std.append(func(sample1, sample2))
        ts_0v100_same_std.append(func(sample2, sample1))

    # Null sample comparisons
    for sample1, sample2 in zip(null_samples_0v1_same_std[0], null_samples_0v1_same_std[1]):
        ts_null_0v1_same_std.append(func(sample1, sample2))
        ts_null_0v1_same_std.append(func(sample2, sample1))

    for sample1, sample2 in zip(null_samples_0v2_same_std[0], null_samples_0v2_same_std[1]):
        ts_null_0v2_same_std.append(func(sample1, sample2))
        ts_null_0v2_same_std.append(func(sample2, sample1))

    for sample1, sample2 in zip(null_samples_0v4_same_std[0], null_samples_0v4_same_std[1]):
        ts_null_0v4_same_std.append(func(sample1, sample2))
        ts_null_0v4_same_std.append(func(sample2, sample1))

    all_dists = (ts_0v1_same_std + ts_0v2_same_std + ts_0v4_same_std +
                 ts_0v10_same_std + ts_0v100_same_std +
                 ts_null_0v1_same_std + ts_null_0v2_same_std + ts_null_0v4_same_std)

    # If the solution produces NaN or inf values OR there are too few unique values OR the evolved test doesn't include both distributions, then throw it out
    if np.any(np.isnan(all_dists)) or np.any(np.isinf(all_dists)) or len(np.unique(all_dists)) < int(0.5 * len(all_dists)) or 'x1' not in str(individual) or 'x2' not in str(individual):
        total_skipped += 1
        solution_cache[str(individual)] = (0., sys.maxsize, sys.maxsize)
        return solution_cache[str(individual)]

    ts_probabilities = []
    for (ts_sig_diff, ts_null) in zip([ts_0v1_same_std, ts_0v2_same_std, ts_0v4_same_std],
                                      [ts_null_0v1_same_std, ts_null_0v2_same_std, ts_null_0v4_same_std]):
        ts_null_kde = scipy.stats.gaussian_kde(ts_null)
        for ts in ts_sig_diff:
            ts = abs(ts)
            prob_score_right_tail = abs(ts_null_kde.integrate_box_1d(ts, float('inf')))
            prob_score_left_tail = abs(ts_null_kde.integrate_box_1d(-float('inf'), -ts))
            # Lower probabilities are better here, so invert the probability
            probability_score = 1. - (prob_score_right_tail + prob_score_left_tail)
            ts_probabilities.append(probability_score)

    for ts_null in [ts_null_0v1_same_std, ts_null_0v2_same_std, ts_null_0v4_same_std]:
        ts_null_kde = scipy.stats.gaussian_kde(ts_null)
        for ts in ts_null:
            ts = abs(ts)
            prob_score_right_tail = abs(ts_null_kde.integrate_box_1d(ts, float('inf')))
            prob_score_left_tail = abs(ts_null_kde.integrate_box_1d(-float('inf'), -ts))
            # Higher probabilities are better here
            probability_score = prob_score_right_tail + prob_score_left_tail
            ts_probabilities.append(probability_score)

    if np.any(np.isnan(ts_probabilities)) or np.any(np.isinf(ts_probabilities)):
        total_skipped += 1
        solution_cache[str(individual)] = (0., sys.maxsize, sys.maxsize)
        return solution_cache[str(individual)]

    # Second fitness component: Test statistic should be scale invariant
    test_stats_invariance = 0.
    for ts_pairs in zip(ts_0v1_same_std, ts_0v10_same_std, ts_0v100_same_std):
        test_stats_invariance += np.std(ts_pairs)

    #for ts_pairs in zip(ts_null_0v1_same_std, ts_null_0v2_same_std, ts_null_0v4_same_std):
    #    test_stats_invariance += np.std(ts_pairs)

    # Third fitness component is the size (i.e., complexity) of the GP tree
    ind_complexity = np.sum([type(component) == gp.Primitive for component in individual])
    #ind_complexity = np.sum([type(component) == gp.Primitive and 'array' in component.name for component in individual])
    
    solution_cache[str(individual)] = (
        round(np.mean(ts_probabilities), 3),
        round(test_stats_invariance, 3),
        ind_complexity
    )
    return solution_cache[str(individual)]

toolbox.register('evaluate', evaluate_individual)
toolbox.register('select', tools.selNSGA2)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('expr_mut', gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def pareto_eq(ind1, ind2):
    """Determines whether two individuals are equal on the Pareto front
    Parameters
    ----------
    ind1: DEAP individual from the GP population
        First individual to compare
    ind2: DEAP individual from the GP population
        Second individual to compare
    Returns
    ----------
    individuals_equal: bool
        Boolean indicating whether the two individuals are equal on
        the Pareto front
    """
    return np.allclose(ind1.fitness.values, ind2.fitness.values)

pop_size = 500

pop = toolbox.population(n=pop_size)
pareto_front = tools.ParetoFront(similar=pareto_eq)
#stats = tools.Statistics(lambda ind: ind.fitness.values[0:3])
stats = tools.Statistics(lambda x: pareto_front)
#stats.register('avg', np.mean, axis=0)
#stats.register('std', np.std, axis=0)
#stats.register('min', np.min, axis=0)
#stats.register('max', np.max, axis=0)
stats.register('num skipped', lambda x: total_skipped)
stats.register('cache lookups', lambda x: total_cache_lookups)
stats.register('pf size', lambda x: len(x[0]))
stats.register('best fitness', lambda x: x[0][0].fitness.values)
stats.register('best ind', lambda x: str(x[0][0]).strip())

t_test = creator.Individual.from_string('float_div(float_sub(array_mean(x1), array_mean(x2)), float_sqrt(float_add(float_div(array_var(x1), array_size(x1)), float_div(array_var(x2), array_size(x2)))))', pset)
print('t-test fitness: {}'.format(evaluate_individual(t_test)))

t_test = creator.Individual.from_string('float_div(float_sub(array_mean(x1), array_mean(x2)), float_add(array_stderr(x1), array_stderr(x2)))', pset)
print('t-test fitness: {}'.format(evaluate_individual(t_test)))

print('')

algorithms.eaMuPlusLambda(population=pop, toolbox=toolbox,
                          cxpb=0.5, mutpb=0.5, mu=pop_size, lambda_=pop_size,
                          ngen=100, stats=stats, halloffame=pareto_front)

print('')
for index, ind in enumerate(pareto_front):
    print(index, ind.fitness, ind)
