import numpy as np
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize
import copy
from verification.utils import massage_proj, LBFs_UBFs_onReLU, LBFs_UBFs_onSigmoid, get_solver, solver_log_filename, lower_bound_from_logs
from time import time

from tqdm import tqdm
import math


def build_individual_problem_MILP(problem,X_n,l,U,model,x,constraintMap,mip,weights=None):
    """
    problem: uninitialised minimisation/maximisation pulp problem object
    X_n: number of input features
    l: proj's eigenvalues
    U: proj's eigenvectors
    model: tensorflow model to verify
    x: data point to verify
    mip: boolean flag - whether to solve the interemediate problems as booleans or as lps

    return: pulp problem with constraints and objective function
    """

    cat_list = ['Continuous' if c == 0 else 'Binary' for c in constraintMap]
    names = ["X_" + str(i) for i in range(len(cat_list))]
    X_i = {i : LpVariable(names[i], lowBound=0, upBound=1, cat = cat_list[i])
                   for i in range(len(cat_list))  }

    # Input constraints

    #Constraints coming from one-hot encoding
    for c in np.unique(constraintMap):
        if c > 0:
            problem += lpSum( [X_i[j] for j in X_n if constraintMap[j] == int(c)]   ) == 1

    for i in X_n:
        if l[i] >= 1e-1:
            bound = 1 / np.sqrt(l[i])
            # Lower Bound
            problem += lpSum(
                    [ U[i,j] * X_i[j] for j in X_n ]) - lpSum([ U[i,j] * x[j] for j in X_n ]) >= - bound
            # Upper Bound
            problem += lpSum(
                    [ U[i,j] * X_i[j] for j in X_n ]) - lpSum([ U[i,j] * x[j] for j in X_n ]) <= bound

    zeta_i = X_i

    if(weights is None):
        iterate_over = model.layers
        num_layers = len(model.layers)
    else:
        iterate_over = []
        for i in range(len(weights)):
            if(i%2 == 0):
               iterate_over.append((np.asarray(weights[i]), np.asarray(weights[i+1])))
        num_layers = int(len(weights)/2)
        #print("ITERATE OVER: ", iterate_over)
        #print("I THINK THESE ARE THE WEIGHTS: ", weights)
        #print("I THINK THERE ARE THIS MANY LAYERS: ", num_layers)
    for i, layer in enumerate(iterate_over):
        #print("INSIDE OF ITERATING OVER")
        if(weights is None):
            W, b = layer.get_weights()
        else:
            W, b = iterate_over[i]
            #print("BOOM WEIGHT: %s, BOOM BIAS %s"%(W, b))

        zeta_n = range(len(zeta_i))
        W_n = range(W.shape[1])
        phi_i  = LpVariable.dicts(f"phi_i_l{i}", (W_n), cat='Continuous')

        # Add affine constraints
        for j, _ in enumerate(phi_i):
            problem += lpSum([ W[k,j] * zeta_i[k] for k in  zeta_n  ]) + b[j]  == phi_i[j]

        # build the Lower and upper bounds
        last_layer = i == num_layers - 1
        if not last_layer:
            phi_i_Ls = []
            phi_i_Us = []
            alpha_Ls = []
            alpha_Us = []
            beta_Ls = []
            beta_Us = []
            zeta_i  = LpVariable.dicts("zeta_i_l" + str(i) , (W_n), cat='Continuous',lowBound=0) #The Relu output is always greater equal than zero
            beta_i  = LpVariable.dicts("beta_i_l" + str(i) , (W_n), cat='Binary',lowBound=0, upBound=1)
            for j, _ in enumerate(phi_i):
                problem_for_phi_i = copy.copy(problem)
                problem_for_phi_i += phi_i[j]
                # Minimisation
                problem_for_phi_i.sense = 1
                problem_for_phi_i.solve(get_solver()(msg = 0, mip = mip))
                phi_i_Ls.append(problem_for_phi_i.objective.value())
                # Maximisation
                problem_for_phi_i.sense = -1
                problem_for_phi_i.solve(get_solver()(msg = 0,mip = mip))
                phi_i_Us.append(problem_for_phi_i.objective.value())

                # Now I compute Relu upper and lower bounding coefficients
                #Relu MILP discretisation comes from: https://arxiv.org/abs/1711.07356
                if phi_i_Us[j] <= 0:
                    problem += (zeta_i[j] == 0)
                elif phi_i_Ls[j] >=0:
                    problem += (zeta_i[j] == phi_i[j])
                else:
                    problem += (zeta_i[j] <= phi_i[j] - phi_i_Ls[j]*(1 - beta_i[j]) )
                    problem += (zeta_i[j] <= phi_i_Us[j]*beta_i[j])
                    problem += (zeta_i[j] >= phi_i[j])
                #[alpha_L,beta_L,alpha_U,beta_U] = LBFs_UBFs_onReLU(phi_i_Ls[j],phi_i_Us[j])
                #alpha_Ls.append(alpha_L)
                #beta_Ls.append(beta_L)
                #alpha_Us.append(alpha_U)
                #beta_Us.append(beta_U)
                #problem += ( zeta_i[j] >= alpha_Ls[j]+ beta_Ls[j]*phi_i[j] )
                #problem += ( zeta_i[j] <= alpha_Us[j]+ beta_Us[j]*phi_i[j] )

            #phi_i_prev = phi_i
            zeta_n = range(len(zeta_i))

    problem += phi_i[0]
    return problem



def build_individual_problem_MILP_with_IBP(problem,X_n,l,U,model,x,constraintMap,mip,weights=None, init_val=None):
    """
    problem: uninitialised minimisation/maximisation pulp problem object
    X_n: number of input features
    l: proj's eigenvalues
    U: proj's eigenvectors
    model: tensorflow model to verify
    x: data point to verify
    mip: boolean flag - whether to solve the interemediate problems as booleans or as lps

    return: pulp problem with constraints and objective function
    """

    cat_list = ['Continuous' if c == 0 else 'Binary' for c in constraintMap]
    names = ["X_" + str(i) for i in range(len(cat_list))]
    X_i = {i : LpVariable(names[i], lowBound=0, upBound=1, cat = cat_list[i])
                   for i in range(len(cat_list))  }

    # Input constraints

    #Constraints coming from one-hot encoding
    #aux = [X_i[j] for j in X_n if constraintMap[j] == 1]



    for c in np.unique(constraintMap):
        if c > 0:
            problem += lpSum( [X_i[j] for j in X_n if constraintMap[j] == int(c)]   ) == 1

    for i in X_n:
        if l[i] >= 1e-1:
            bound = 1 / np.sqrt(l[i])
            # Lower Bound
            problem += lpSum(
                    [ U[i,j] * X_i[j] for j in X_n ]) - lpSum([ U[i,j] * x[j] for j in X_n ]) >= - bound
            # Upper Bound
            problem += lpSum(
                    [ U[i,j] * X_i[j] for j in X_n ]) - lpSum([ U[i,j] * x[j] for j in X_n ]) <= bound

    zeta_i = X_i
    phi_i_Ls = np.zeros(len(X_i))
    phi_i_Us = np.ones(len(X_i))

    if(weights is None):
        iterate_over = model.layers
        num_layers = len(model.layers)
    else:
        iterate_over = []
        for i in range(len(weights)):
            if(i%2 == 0):
               iterate_over.append((np.asarray(weights[i]), np.asarray(weights[i+1])))
        num_layers = int(len(weights)/2)

    for i, layer in enumerate(iterate_over):
        if(weights is None):
            W, b = layer.get_weights()
        else:
            W, b = iterate_over[i]

        zeta_n = range(len(zeta_i))
        W_n = range(W.shape[1])
        phi_i  = LpVariable.dicts(f"phi_i_l{i}", (W_n), cat='Continuous')

        # Add affine constraints
        for j, _ in enumerate(phi_i):
            problem += lpSum([ W[k,j] * zeta_i[k] for k in  zeta_n  ]) + b[j]  == phi_i[j]

        # build the Lower and upper bounds
        last_layer = i == num_layers - 1
        if not last_layer:
            #phi_i_Ls = []
            #phi_i_Us = []
            alpha_Ls = []
            alpha_Us = []
            beta_Ls = []
            beta_Us = []
            zeta_i  = LpVariable.dicts("zeta_i_l" + str(i) , (W_n), cat='Continuous',lowBound=0) #The Relu output is always greater equal than zero
            beta_i  = LpVariable.dicts("beta_i_l" + str(i) , (W_n), cat='Binary',lowBound=0, upBound=1)

            mu = (phi_i_Us + phi_i_Ls) / 2
            r = (phi_i_Us - phi_i_Ls) / 2

            mu_new = W.T @ mu + b # layer(mean)  should also work
            r_new = np.abs(W.T) @ r
            phi_i_Us = mu_new + r_new
            phi_i_Ls = mu_new - r_new

            for j, _ in enumerate(phi_i):
                #problem_for_phi_i = copy.copy(problem)
                #problem_for_phi_i += phi_i[j]
                # Minimisation
                #problem_for_phi_i.sense = 1
                #problem_for_phi_i.solve(get_solver()(msg = 0, mip = mip))
                #phi_i_Ls.append(problem_for_phi_i.objective.value())
                # Maximisation
                #problem_for_phi_i.sense = -1
                #problem_for_phi_i.solve(get_solver()(msg = 0,mip = mip))
                #phi_i_Us.append(problem_for_phi_i.objective.value())

                # Now I compute Relu upper and lower bounding coefficients
                #Relu MILP discretisation comes from: https://arxiv.org/abs/1711.07356
                if phi_i_Us[j] <= 0:
                    problem += (zeta_i[j] == 0)
                elif phi_i_Ls[j] >=0:
                    problem += (zeta_i[j] == phi_i[j])
                else:
                    problem += (zeta_i[j] <= phi_i[j] - phi_i_Ls[j]*(1 - beta_i[j]) )
                    problem += (zeta_i[j] <= phi_i_Us[j]*beta_i[j])
                    problem += (zeta_i[j] >= phi_i[j])
                #[alpha_L,beta_L,alpha_U,beta_U] = LBFs_UBFs_onReLU(phi_i_Ls[j],phi_i_Us[j])
                #alpha_Ls.append(alpha_L)
                #beta_Ls.append(beta_L)
                #alpha_Us.append(alpha_U)
                #beta_Us.append(beta_U)
                #problem += ( zeta_i[j] >= alpha_Ls[j]+ beta_Ls[j]*phi_i[j] )
                #problem += ( zeta_i[j] <= alpha_Us[j]+ beta_Us[j]*phi_i[j] )
            phi_i_Us = my_relu(phi_i_Us)
            phi_i_Ls = my_relu(phi_i_Ls)
            #phi_i_prev = phi_i
            zeta_n = range(len(zeta_i))

    problem += phi_i[0]
    #print("AYO HERE IS THE FEATURE FOR PROBLEM BUILD 😘: ", X_n)
    #if(init_val is not None):
    #    for ind, key in enumerate(X_i):
    #        #print(init_val, init_val[ind], init_val.shape)
    #        X_i[key].setInitialValue(np.squeeze(init_val[ind]))
    return problem



def verify_locally(test_ds, model, proj, epsilon, delta, opt_mode, time_limit, mip = False ,num_of_points = 5 ):
    ''' Local verification, this could be used as a building block for adversarial learning '''
    print("CHECK THIS ***************: ", type(test_ds))
    X = test_ds.X_df.values
    constraintMap = np.zeros(proj.shape[0])
    for i, category in  enumerate(test_ds.cat_cols):
         #print(i)
        if i < len(test_ds.cat_cols):
            for idx in test_ds.columns_map[category]:
                constraintMap[idx] = i+1

    verifieds = 0
    verification_X = X if num_of_points is None else X[:num_of_points]
    phi_i_lb_vec = []
    phi_i_ub_vec = []
    zeta_lb_vec = []
    zeta_ub_vec = []
    problem_build_time_vec = []
    problem_solve_time_vec = []
    l, U = np.linalg.eigh(proj/(epsilon**2))
    X_n = range(proj.shape[0])
    for i, x in enumerate(verification_X):
        #if i % 10 == 0:
        print(f'Local verification for data point {i}')
        print("Here is the input: ", x)
        #In the following we first solve the problem for the minimum and then the problem for the maximum


        #---MINIMUM---#
        s_t = time()
        problem_min = LpProblem('fairness_constraints', LpMinimize)

        problem_min = build_individual_problem_MILP(
                problem_min, X_n, l, U, model, x, constraintMap,mip)
        e_t = time()
        problem_build_time_vec.append(e_t - s_t)

        s_t = time()
        problem_min.solve(get_solver()(
            msg=0, timeLimit=time_limit, logPath=SOLVER_LOG_FILENAME))
        e_t = time()
        problem_solve_time_vec.append(e_t - s_t)
        phi_i_lb = problem_min.objective.value()
        optimal_point_lb = np.zeros(x.shape)
        optimals_alphabetical_lb = [var for var in problem_min.variables() if var.name[:2] == 'X_' ]
        for entry in optimals_alphabetical_lb:
            idx = int(entry.name[2:])
            optimal_point_lb[idx] = entry.varValue
        #optimal_point_lb = [var.varValue for var in problem_min.variables()]


        #---MAXIMUM---#
        #problem_max = LpProblem('fairness_constraints', LpMaximize)
        #THIS IS EXACTLY THE SAME PROBLEM AND WE COULD AVOID RE-BUILDING IT
        #if opt_mode == 'lp':
        #    problem_max = build_individual_problem(problem_max, X_n, l, U, model, x)
        #elif opt_mode == 'milp':
        #    problem_max = build_individual_problem_MILP(
        #        problem_max, X_n, l, U, model, x, constraintMap,mip)
        problem_max = copy.copy(problem_min)
        problem_max.sense = -1 # setting the maximisation flag explicitly
        problem_max.solve(get_solver()(
            msg=0, timeLimit=time_limit, logPath=SOLVER_LOG_FILENAME))
        phi_i_ub = problem_max.objective.value()
        optimal_point_ub = np.zeros(x.shape)
        optimals_alphabetical_ub = [var for var in problem_min.variables() if var.name[:2] == 'X_' ]
        for entry in optimals_alphabetical_ub:
            idx = int(entry.name[2:])
            optimal_point_ub[idx] = entry.varValue

        zeta_lb = model.layers[-1].activation(phi_i_lb).numpy()
        zeta_ub = model.layers[-1].activation(phi_i_ub).numpy()
        if zeta_ub - zeta_lb <= delta:
            verifieds = verifieds + 1
        phi_i_lb_vec.append(phi_i_lb)
        phi_i_ub_vec.append(phi_i_ub)
        zeta_lb_vec.append(zeta_lb)
        zeta_ub_vec.append(zeta_ub)
        print("Here is the worst case: ", optimal_point_ub)


    logit_bounds = []
    probability_bounds = []
    for i in range(num_of_points):
        logit_bounds.append([phi_i_lb_vec[i],phi_i_ub_vec[i]])
        probability_bounds.append([zeta_lb_vec[i],zeta_ub_vec[i]])

    logit_bounds = np.array(logit_bounds)
    prob_bounds = np.array(probability_bounds)
    logit_diffs = logit_bounds[:,1] - logit_bounds[:,0]
    prob_diffs = prob_bounds[:,1] - prob_bounds[:,0]

    verification_results = {
        'verified_fraction': verifieds/num_of_points,
        'mean_logit_diff': np.mean(logit_diffs),
        'std_logit_diff': np.std(logit_diffs),
        'min_logit_diff': np.min(logit_diffs),
        'max_logit_diff': np.max(logit_diffs),
        'mean_prob_diff': np.mean(prob_diffs),
        'std_prob_diff': np.std(prob_diffs),
        'min_prob_diff': np.min(prob_diffs),
        'max_prob_diff': np.max(prob_diffs),
        'mean_problem_build_time': np.mean(problem_build_time_vec),
        'mean_problem_solve_time': np.mean(problem_solve_time_vec),
        'mean_verification_time': np.mean(problem_build_time_vec + problem_solve_time_vec),
    }
    print("Here is the local verirication result: ", verification_results)

    return verification_results


def my_relu(x):
    return  np.maximum(x,0)


def multiproc_veri(args):
    ''' Local verification, this could be used as a building block for adversarial learning '''
    ''' Lets see how much I can shove in here '''

    (test_ds, batch_idx,  proj, epsilon, delta, opt_mode, time_limit, mip, weights, init_val, proc_id) = args

    X = test_ds.X_df.values
    l, U = np.linalg.eigh(proj/(epsilon**2))
    U = U.T
    X_n = range(proj.shape[0])

    PROC_ID = batch_idx[0]
    constraintMap = np.zeros(proj.shape[0])
    for i, category in  enumerate(test_ds.cat_cols):
        if i < len(test_ds.cat_cols):
            for idx in test_ds.columns_map[category]:
                constraintMap[idx] = i+1

    verifieds = 0
    verification_X = X[batch_idx] #X if num_of_points is None else X[:num_of_points]
    total_inputs = len(batch_idx)
    phi_i_lb_vec = []
    phi_i_ub_vec = []
    zeta_lb_vec = []
    zeta_ub_vec = []
    problem_build_time_vec = []
    problem_solve_time_vec = []
    worst_case_inputs = []
    i = 0
    for x in tqdm(verification_X, desc="Verifying (from proc [%s])"%(PROC_ID)):
        #---MINIMUM---#
        s_t = time()
        problem_min = LpProblem('fairness_constraints', LpMinimize)

        if opt_mode == 'lp':
            problem_min = build_individual_problem(problem_min, X_n, l, U, None, x)
        elif opt_mode == 'milp':
            if(init_val is not None):
                try:
                    problem_min = build_individual_problem_MILP_with_IBP(
                        problem_min, X_n, l, U, None, x, constraintMap,mip, weights=weights, init_val=init_val[i])
                except Exception as e:
                    with open('BUILD_FAIL_STACKTRACE.txt', 'a') as f:
                        f.write(str(e))
                        f.write(traceback.format_exc())
            else:
                try:
                    problem_min = build_individual_problem_MILP_with_IBP(
                        problem_min, X_n, l, U, None, x, constraintMap,mip, weights=weights, init_val=None)
                except Exception as e:
                    with open('BUILD_FAIL_STACKTRACE.txt', 'a') as f:
                        f.write(str(e))
                        f.write(traceback.format_exc())

        e_t = time()
        problem_build_time_vec.append(e_t - s_t)

        s_t = time()
        try:
            problem_min.solve(get_solver()(
                msg=0, timeLimit=18.0, logPath=solver_log_filename(proc_id)))
        except Exception as e:
            with open('SOLVE_FAIL_STACKTRACE.txt', 'a') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
        e_t = time()

        try:
            resulting_lb = problem_min.objective.value()
            log_value = lower_bound_from_logs(proc_id)
            phi_i_lb = resulting_lb
            if phi_i_lb is None:
                phi_i_lb = log_value
            if phi_i_lb is None:
                raise ValueError(f'Got None value from lower_bound_from_logs: {phi_i_lb}')
        except Exception as e:
            with open('PROCESSING_FAIL_STACKTRACE.txt', 'a') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
        optimal_point_lb = np.zeros(x.shape)
        optimals_alphabetical_lb = [var for var in problem_min.variables() if var.name[:2] == 'X_' ]
        for entry in optimals_alphabetical_lb:
            idx = int(entry.name[2:])
            optimal_point_lb[idx] = entry.varValue


        problem_max = copy.copy(problem_min)
        problem_max.sense = -1 # setting the maximisation flag explicitly
        try:
            problem_max.solve(get_solver()(
                msg=0, timeLimit=18.0, logPath=solver_log_filename(proc_id)))
        except Exception as e:
            with open('SOLVE_FAIL_STACKTRACE.txt', 'a') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
        try:
            resulting_ub = problem_max.objective.value()
            log_value = lower_bound_from_logs(proc_id)
            phi_i_ub = resulting_ub
            if phi_i_ub is None:
                phi_i_ub = log_value
            if phi_i_ub is None:
                raise ValueError(f'Got None value from lower_bound_from_logs: {phi_i_ub}')
        except Exception as e:
            with open('PROCESSING_FAIL_STACKTRACE.txt', 'a') as f:
                f.write(str(e))
                f.write(traceback.format_exc())


        optimal_point_ub = np.zeros(x.shape)
        optimals_alphabetical_ub = [var for var in problem_min.variables() if var.name[:2] == 'X_' ]
        for entry in optimals_alphabetical_ub:
            idx = int(entry.name[2:])
            optimal_point_ub[idx] = entry.varValue

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        zeta_lb = sigmoid(phi_i_lb)
        zeta_ub = sigmoid(phi_i_ub)

        if zeta_ub - zeta_lb <= delta:
            verifieds = verifieds + 1
        phi_i_lb_vec.append(phi_i_lb)
        phi_i_ub_vec.append(phi_i_ub)
        zeta_lb_vec.append(zeta_lb)
        zeta_ub_vec.append(zeta_ub)
        worst_case_inputs.append(optimal_point_ub)
        i += 1

    return np.asarray(worst_case_inputs)

