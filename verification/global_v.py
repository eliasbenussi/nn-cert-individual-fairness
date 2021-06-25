from time import time
import numpy as np
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize
import copy
from verification.utils import massage_proj, LBFs_UBFs_onReLU, LBFs_UBFs_onSigmoid, solver_log_filename, lower_bound_from_logs, my_sigmoid, discretise_sigmoid_interval, get_solver

from matplotlib import pyplot as plt


def _set_input_constraints(problem, X_n, X_i, X_ii, l, U, model):
    for i in X_n:
        if l[i] >= 1e-1:
            bound = 1 / np.sqrt(l[i])
            # Lower Bound
            problem += lpSum(
                    [ U[i,j] * X_i[j] for j in X_n ]) - lpSum([ U[i,j] * X_ii[j] for j in X_n ]) >= - bound
            # Upper Bound
            problem += lpSum(
                    [ U[i,j] * X_i[j] for j in X_n ]) - lpSum([ U[i,j] * X_ii[j] for j in X_n ]) <= bound


def _set_affine_constraints(problem, W, b, phi_i, phi_ii, zeta_i, zeta_ii, zeta_n):
    for j, _ in enumerate(phi_i):
        problem += lpSum([ W[k,j] * zeta_i[k] for k in zeta_n ]) + b[j]  == phi_i[j]
        problem += lpSum([ W[k,j] * zeta_ii[k] for k in zeta_n ]) + b[j]  == phi_ii[j]


def build_global_problem_on_confidence_difference_MILP(problem, X_n, l, U, model, constraintMap, M, mip, time_limit):
    ''' Work in Progress'''

    # Generate constraints for categorical features
    cat_list = ['Continuous' if c == 0 else 'Binary' for c in constraintMap]
    names = ["X_i_" + str(i) for i in range(len(cat_list))]
    X_i = {i:LpVariable(names[i], lowBound=0, upBound=1, cat = cat_list[i])
                   for i in range(len(cat_list))  }
    names = ["X_ii_" + str(i) for i in range(len(cat_list))]
    X_ii = {i:LpVariable(names[i], lowBound=0, upBound=1, cat = cat_list[i])
                   for i in range(len(cat_list))  }

    for c in np.unique(constraintMap):
        if c > 0:
            problem += lpSum( [X_i[j]  for j in X_n if constraintMap[j] == int(c)]   ) == 1
            problem += lpSum( [X_ii[j] for j in X_n if constraintMap[j] == int(c)]   ) == 1

    # Input constraints
    _set_input_constraints(problem, X_n, X_i, X_ii, l, U, model)

    zeta_i = X_i
    zeta_ii = X_ii
    for i, layer in enumerate(model.layers):
        zeta_n = range(len(zeta_i))
        W, b = layer.get_weights()
        W_n = range(W.shape[1])
        phi_i  = LpVariable.dicts(f"phi_i_l{i}", (W_n), cat='Continuous')
        phi_ii =  LpVariable.dicts(f"phi_ii_l{i}", (W_n), cat='Continuous')

        # Add affine constraints
        _set_affine_constraints(problem, W, b, phi_i, phi_ii, zeta_i, zeta_ii, zeta_n)

        # build the Lower and upper bounds
        last_layer = i == len(model.layers) - 1

        phi_i_Ls = []
        phi_i_Us = []
        alpha_Ls = []
        alpha_Us = []
        beta_Ls = []
        beta_Us = []
        if not last_layer:
            #The Relu output is always greater equal than zero
            zeta_i  = LpVariable.dicts("zeta_i_l" + str(i) , (W_n), cat='Continuous',lowBound=0)
            zeta_ii  = LpVariable.dicts("zeta_ii_l" + str(i) , (W_n), cat='Continuous',lowBound=0)

            beta_i  = LpVariable.dicts("beta_i_l" + str(i) , (W_n), cat='Binary',lowBound=0, upBound=1)
            beta_ii  = LpVariable.dicts("beta_ii_l" + str(i) , (W_n), cat='Binary',lowBound=0, upBound=1)

        for j in phi_i:
            problem_for_phi_i = copy.copy(problem)
            problem_for_phi_i += phi_i[j]

            # Minimisation
            problem_for_phi_i.sense = 1
            problem_for_phi_i.solve(get_solver()(msg = 0, mip=mip))
            phi_i_Ls.append(problem_for_phi_i.objective.value())

            # Maximisation
            problem_for_phi_i.sense = -1
            problem_for_phi_i.solve(get_solver()(msg = 0, mip=mip))
            phi_i_Us.append(problem_for_phi_i.objective.value())

            # Now I compute Relu upper and lower bounding coefficients
            if not last_layer:

                # We are in the flat side of the Relu
                if phi_i_Us[j] <= 0:
                    problem += (zeta_i[j] == 0)
                    problem += (zeta_ii[j] == 0)

                # We are in the identity side of the Relu
                elif phi_i_Ls[j] >=0:
                    problem += (zeta_i[j] == phi_i[j])
                    problem += (zeta_ii[j] == phi_ii[j])

                else:
                    problem += (zeta_i[j] <= phi_i[j] - phi_i_Ls[j]*(1 - beta_i[j]) )
                    problem += (zeta_i[j] <= phi_i_Us[j]*beta_i[j])
                    problem += (zeta_i[j] >= phi_i[j])

                    problem += (zeta_ii[j] <= phi_ii[j] - phi_i_Ls[j]*(1 - beta_ii[j]) )
                    problem += (zeta_ii[j] <= phi_i_Us[j]*beta_ii[j])
                    problem += (zeta_ii[j] >= phi_ii[j])
            else:
                assert(j==0)

                #NEW IMPLEMENTATION FOR PHI_GRID BELOW. The idea is that now it is adaptive to the shape of the sigmoid, instead of uniform.
                if (phi_i_Us[j] <= 0) or (phi_i_Ls[j] >= 0): #we are in the part of the sigmoid in which the convexity doesn't change
                    phi_grid = discretise_sigmoid_interval(2*M+1,my_sigmoid(phi_i_Ls[j]),my_sigmoid(phi_i_Us[j]) )
                else:
                    phi_grid = np.concatenate((discretise_sigmoid_interval(M+1,my_sigmoid(phi_i_Ls[j]),0.5),discretise_sigmoid_interval(M+1,0.5,my_sigmoid(phi_i_Us[j]))[1:] ))

                #To do so, I First have to iterate over the various intervals
                y_grid_lb = []
                y_grid_ub = []
                prev_value_lb = np.inf
                prev_value_ub = - np.inf
                for i_grid in range(len(phi_grid) - 1):
                    curr_x_L = phi_grid[i_grid]
                    curr_x_U = phi_grid[i_grid + 1]
                    [alpha_L,beta_L,alpha_U,beta_U] = LBFs_UBFs_onSigmoid(curr_x_L,curr_x_U)
                    y_grid_lb.append(min(alpha_L + beta_L*curr_x_L, prev_value_lb))
                    y_grid_ub.append(max(alpha_U + beta_U*curr_x_L, prev_value_ub))
                    prev_value_lb = alpha_L + beta_L*curr_x_U
                    prev_value_ub = alpha_U + beta_U*curr_x_U
                y_grid_lb.append(prev_value_lb)
                y_grid_ub.append(prev_value_ub)


                #binary value for activation of piecewise component
                y_n = range(2*M)
                lambda_n = range(2*M+1)
                y_i = LpVariable.dicts(f"y_i_{i}", (y_n), cat='Binary',lowBound=0, upBound=1)
                y_ii = LpVariable.dicts(f"y_ii_{i}", (y_n), cat='Binary',lowBound=0, upBound=1)
                lambda_i = LpVariable.dicts("lambda_i_" + str(i) , (lambda_n), cat='Continuous',lowBound=0, upBound=1)
                lambda_ii = LpVariable.dicts("lambda_ii_" + str(i) , (lambda_n), cat='Continuous',lowBound=0, upBound=1)

                #constraints on y_i encoding
                problem += lpSum( y_i[k]  for k in y_n ) == 1
                problem += lpSum( y_ii[k]  for k in y_n ) == 1
                problem += lpSum( lambda_i[k]  for k in lambda_n ) == 1
                problem += lpSum( lambda_ii[k]  for k in lambda_n ) == 1

                #constraints on lambda parameters
                for k in y_n:
                    problem += ( y_i[k]  <= lambda_i[k]  + lambda_i[k+1])
                    problem += ( y_ii[k] <= lambda_ii[k] + lambda_ii[k+1])

                problem += lpSum( phi_grid[k]*lambda_i[k]  for k in lambda_n ) ==  phi_i[j]
                problem += lpSum( phi_grid[k]*lambda_ii[k] for k in lambda_n  ) ==  phi_ii[j]

                # Optimisation objective
                problem += lpSum( y_grid_ub[k]*lambda_i[k] - y_grid_lb[k]*lambda_ii[k] for k in lambda_n)

    return problem


def verify_globally(test_ds, model, U, l, epsilon, delta, opt_mode, M, time_limit, mip, proc_id):
    ''' Implementing global property for the logit case - will have to see how to extend
        it to the actual classification confidence level later...'''

    print(f'\n\nSOLVER LOG FILE: {solver_log_filename(proc_id)}\n\n')

    X_n = range(test_ds.X_df.values.shape[1])
    constraintMap = np.zeros(test_ds.X_df.values.shape[1])
    for i, category in  enumerate(test_ds.cat_cols):
        if i < len(test_ds.cat_cols):
            for idx in test_ds.columns_map[category]:
                constraintMap[idx] = i+1

    print('Building Max Confidence difference problem')
    s_t = time()
    problem_max = LpProblem('fairness_constraints', LpMaximize)
    if opt_mode == 'lp':
        problem_max = build_global_problem_on_confidence_difference_lp(
            problem_max, X_n, l, U, model)
    elif opt_mode == 'milp':
        problem_max = build_global_problem_on_confidence_difference_MILP(
            problem_max, X_n, l, U, model, constraintMap, M, mip=False, time_limit=time_limit/5)
    e_t = time()
    problem_build_time = e_t - s_t
    print(f'---> Time passed building constraints: {problem_build_time}')

    print('Solving Max Confidence difference problem')
    s_t = time()
    problem_max.solve(get_solver()(
        mip=mip, timeLimit=time_limit, logPath=solver_log_filename(proc_id)))
    e_t = time()
    problem_solve_time = e_t - s_t
    print(f'---> Time passed solving constraints: {problem_solve_time}')

    prob_value = problem_max.objective.value()
    log_value = lower_bound_from_logs(proc_id)
    print(f'\nProblem Value is: {prob_value}')
    print(f'\nLog Value is: {log_value}')
    max_confidence_difference = prob_value
    if prob_value is None:
        max_confidence_difference = log_value
    print(f'Max Confidence Diff is: {max_confidence_difference}')

    #The problem is completely symmetric on x' and x'', so the minimum is just the opposite value of the maximum.
    verifieds = float(max_confidence_difference <= delta)

    verification_results = {
        'verified_fraction': verifieds,
        'max_confidence_diff': max_confidence_difference,
        'problem_build_time': problem_build_time,
        'problem_solve_time': problem_solve_time,
        'verification_time': problem_build_time + problem_solve_time,
    }

    return verification_results


def verify_globally_for_weighted_lp(test_ds, model, epsilon, delta, opt_mode, M, time_limit, mip, proc_id):
    U = np.eye(test_ds.X_df.values.shape[1]) / epsilon
    l = np.ones(test_ds.X_df.values.shape[1])
    return verify_globally(
        test_ds, model, U, l, epsilon, delta, opt_mode, M, time_limit, mip, proc_id)


def verify_globally_for_mahalanobis(test_ds, model, proj, epsilon, delta, opt_mode, M, time_limit, mip, proc_id):
    l, U = np.linalg.eigh(proj/(epsilon**2))
    U = U.T
    return verify_globally(
        test_ds, model, U, l, epsilon, delta, opt_mode, M, time_limit, mip, proc_id)

