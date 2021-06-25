import os
import numpy as np
from pulp import PULP_CBC_CMD, GUROBI


SOLVER_LOG_FILENAME = './.solver-log.log'
SOLVER_VAR = 'FAIR_SOLVER'
DEFAULT_SOLVER = 'GUROBI'


def solver_log_filename(proc_id):
    return f'/tmp/.solver-log-{proc_id}.log'


def lower_bound_from_logs(proc_id):
    time_limit_reached = False
    lb = None

    with open(solver_log_filename(proc_id), "r") as f:
        for line in f:
            cbc_stop_marker = 'Result - Stopped on time limit'
            gurobi_stop_marker = 'Result - User ctrl-cuser ctrl-c'
            if (cbc_stop_marker in line) or (gurobi_stop_marker in line):
                print('\nFOUND MARKER IN LOGFILE\n')
                time_limit_reached = True

            lower_bound_marker = 'Lower bound'
            if lower_bound_marker in line and time_limit_reached:
                lb = float(line.split(' ')[-1])
                print(f'\nFOUND LOWER BOUND IN LOGFILE: {lower_bound_marker}\n', lb)
    return lb

def get_solver():
    fair_solver = os.getenv(SOLVER_VAR)
    if fair_solver == 'GUROBI':
        return GUROBI
    else:
        return PULP_CBC_CMD


def massage_proj(proj, jitter=0.1):
    """
    proj: projection matrix of non sensitive directions
    epsilon: massaging factor

    return: a matrix shifted to not have zeros along the diagonal
    """
    proj = proj.copy()
    shift = jitter * np.identity(proj.shape[0])
    return proj + shift


def LBFs_UBFs_onReLU(phi_l, phi_u):
    alpha_u, beta_u = line_thru_points(phi_l, my_relu(phi_l), phi_u,my_relu(phi_u))

    # TODO: I feel like this could just be:

    # alpha_L,beta_L = 0, beta_U

    # since the lower bound has the same angular coeff and 0 intercept?
    # This is already assuming we are using ReLU, so I feel like this generalisation doesn't
    # add a lot of flexibility, especially since I don't believe this is the logic we would use for
    # e.g. sigmoid?
    alpha_l, beta_l = line_thru_point_with_ang_coeff(0,0,beta_u)
    return alpha_l, beta_l, alpha_u, beta_u


def LBFs_UBFs_onSigmoid(phi_i_L,phi_i_U):
    #The sigmoid is convex on the left part, concave in the right part and has a single flex point in 0
    flex_point = 0

    if phi_i_U <= flex_point: #in this case we just need to care about the convex bit
        [alpha_L, beta_L, alpha_U, beta_U] = convex_bounds(phi_i_L,phi_i_U, my_sigmoid, my_sigmoid_derivative)
    elif phi_i_L >= flex_point: #in this case we just need to care about the concave bit
        [alpha_L, beta_L, alpha_U, beta_U] = concave_bounds(phi_i_L,phi_i_U, my_sigmoid, my_sigmoid_derivative)
    else:
        # in this case we have to do all the computations and then merge the LBFs and UBFs that we find
        [alpha_L_cvx, beta_L_cvx, alpha_U_cvx, beta_U_cvx] = convex_bounds(phi_i_L,0, my_sigmoid, my_sigmoid_derivative)
        [alpha_L_cnc, beta_L_cnc, alpha_U_cnc, beta_U_cnc] = concave_bounds(0,phi_i_U, my_sigmoid, my_sigmoid_derivative)
        [alpha_L, beta_L] = merge_LBFs(alpha_L_cnc, beta_L_cnc,alpha_L_cvx, beta_L_cvx,phi_i_L,phi_i_U,'cvx_cnc')
        [alpha_U, beta_U] = merge_UBFs(alpha_U_cnc, beta_U_cnc,alpha_U_cvx, beta_U_cvx,phi_i_L,phi_i_U,'cvx_cnc')

    return alpha_L, beta_L, alpha_U, beta_U


def my_relu(x):
    return  max(x,0)


def my_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def my_sigmoid_derivative(x):
    return my_sigmoid(x)*(1- my_sigmoid(x))


def discretise_sigmoid_interval(M,init_logit,fin_logit):
    return logit_fun(np.linspace(init_logit,fin_logit,M))


def  logit_fun(p):
    p = np.clip(p, 0.000001, 0.999999) # clip to avoid numerical errors
    return np.log( p/(1-p)  )


def  concave_bounds(x_l,x_u, fun, d_fun ):
    f_l = fun(x_l)
    f_u = fun(x_u)
    [a_lower,b_lower] = line_thru_points(x_l,f_l,x_u,f_u)
    x_c = 0.5*(x_l + x_u)
    df_c = d_fun(x_c)
    f_c = fun(x_c)
    [a_upper,b_upper] = line_thru_point_with_ang_coeff(x_c,f_c , df_c)
    return [a_lower,b_lower,a_upper,b_upper]

def convex_bounds(x_l,x_u, fun, d_fun):

    #LBF computations
    x_c = 0.5*(x_l + x_u)
    df_c = d_fun(x_c)
    f_c = fun(x_c)
    [a_lower,b_lower] = line_thru_point_with_ang_coeff(x_c,f_c , df_c)
    #UBF computations
    f_l = fun(x_l)
    f_u = fun(x_u)
    [a_upper,b_upper] = line_thru_points(x_l,f_l,x_u,f_u)
    return [a_lower,b_lower,a_upper,b_upper]


def line_thru_points(x_l, f_l, x_u, f_u):
    if x_l < x_u:
        beta = (f_l - f_u) / (x_l - x_u)
        alpha = f_l - beta * x_l
    else:
        beta = 0
        alpha = f_l
    return alpha, beta


def line_thru_point_with_ang_coeff(x, f_x, df_x):
    beta = df_x
    alpha = - beta * x + f_x
    return alpha, beta


def merge_LBFs(a_cnc,b_cnc,a_cvx,b_cvx,x_l,x_u,merging_sense):
    if merging_sense == 'cvx_cnc':
        f_l = a_cvx + b_cvx * x_l

        f1_u = a_cnc + b_cnc * x_u
        f2_u = a_cvx + b_cvx * x_u
        f_u = min(f1_u,f2_u)
    elif merging_sense == 'cnc_cvx':
        f1_l = a_cnc + b_cnc * x_l
        f2_l = a_cvx + b_cvx * x_l

        f_l = min(f1_l,f2_l)
        f_u = a_cvx + b_cvx * x_u
    [a,b] = line_thru_points(x_l,f_l,x_u,f_u)

    return a, b


def merge_UBFs(a_cnc,b_cnc,a_cvx,b_cvx,x_l,x_u,merging_sense):

    if merging_sense == 'cvx_cnc':
        g_u = a_cnc + b_cnc * x_u

        g1_l = a_cnc + b_cnc * x_l
        g2_l = a_cvx + b_cvx * x_l
        g_l = max(g1_l,g2_l)
    elif merging_sense == 'cnc_cvx':
        g_l =  a_cnc + b_cnc * x_l

        g1_u = a_cnc + b_cnc * x_u
        g2_u = a_cvx + b_cvx * x_u

        g_u = max(g1_u,g2_u)

    [a,b] = line_thru_points(x_l,g_l,x_u,g_u);
    return a, b
