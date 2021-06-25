import os
from time import time
from datetime import datetime
import uuid
import numpy as np
from sklearn.decomposition import TruncatedSVD
import tensorflow.compat.v1 as tf1
from collections import OrderedDict
from dataset.biased_dataset import BiasedDataset

import tensorflow as tf2
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool

GLOBAL_KERAS_MODEL = None
GLOBAL_NPROC = 10 #if cpu_count() > 10 else 4


import sys
sys.path.append("../verification")
from verification.local_v import multiproc_veri

def compl_svd_projector(names, svd=-1):
    if svd > 0:
        tSVD = TruncatedSVD(n_components=svd)
        tSVD.fit(names)
        basis = tSVD.components_.T
        print('Singular values:')
        print(tSVD.singular_values_)
    else:
        basis = names.T

    proj = np.linalg.inv(np.matmul(basis.T, basis))
    proj = np.matmul(basis, proj)
    proj = np.matmul(proj, basis.T)
    proj_compl = np.eye(proj.shape[0]) - proj
    return proj_compl


def fair_dist(proj, w=0.):
    tf_proj = tf1.constant(proj, dtype=tf1.float32)
    if w>0:
        return lambda x, y: tf1.reduce_sum(tf1.square(tf1.matmul(x-y,tf_proj)) + w*tf1.square(tf1.matmul(x-y,tf1.eye(proj.shape[0]) - tf_proj)), axis=1)
    else:
        return lambda x, y: tf1.reduce_sum(tf1.square(tf1.matmul(x-y,tf_proj)), axis=1)


def weight_variable(shape, name):
    if len(shape)>1:
        init_range = np.sqrt(6.0/(shape[-1]+shape[-2]))
    else:
        init_range = np.sqrt(6.0/(shape[0]))
    initial = tf1.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf1.float32) # seed=1000
    return tf1.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf1.constant(0.1, shape=shape)
    return tf1.Variable(initial, name=name)


def sample_batch_idx_binary(y, n_per_class):
    batch_idx = []
    batch_idx += np.random.choice(np.where(y[:,0]==1)[0], size=n_per_class, replace=False).tolist()
    batch_idx += np.random.choice(np.where(y[:,0]==0)[0], size=n_per_class, replace=False).tolist()
    np.random.shuffle(batch_idx)
    return batch_idx


def sample_batch_idx_regression(y, n_samples):
    batch_idx = np.random.choice(np.array(range(y.shape[0])), size=n_samples, replace=False)
    np.random.shuffle(batch_idx)
    return batch_idx


def fc_network(variables, layer_in, n_layers, l=0, activ_f = tf1.nn.relu, units = []):
    if l==n_layers-1:
        layer_out = tf1.matmul(layer_in, variables['weight_'+str(l)]) + variables['bias_' + str(l)]
        units.append(layer_out)
        class_probs = activ_f(layer_out)
        return layer_out, units, class_probs
    else:
        layer_out = activ_f(tf1.matmul(layer_in, variables['weight_'+str(l)]) + variables['bias_' + str(l)])
        l += 1
        units.append(layer_out)
        return fc_network(variables, layer_out, n_layers, l=l, activ_f=activ_f, units=units)


def forward_fair_binary(tf_X, tf_y, tf_adv_X, lam, n_units = None, activ_f = tf1.nn.relu, l2_reg=0., goal='classification'):

    n_features = int(tf_X.shape[1])
    n_class = int(tf_y.shape[1])
    n_layers = len(n_units) + 1
    n_units = [n_features] + n_units + [n_class]

    variables = OrderedDict()
    for l in range(n_layers):
        variables['weight_' + str(l)] = weight_variable(
            [n_units[l],n_units[l+1]], name='weight_' + str(l))
        variables['bias_' + str(l)] = bias_variable([n_units[l+1]], name='bias_' + str(l))

    ## Defining a global keras model for the MILP problem
    model = Sequential()
    for _ in range(n_layers-1):
        if(_ == 0):
            model.add(Dense(int(n_units[_+1]), activation='relu', input_shape=(int(n_features),)))
        else:
            model.add(Dense(n_units[_+1], activation='relu'))
    model.add(Dense(n_units[-1], activation='sigmoid'))

    metrics = []
    if goal == 'classification':
        metrics = [
            tf2.keras.metrics.TruePositives(name='tp'),
            tf2.keras.metrics.FalsePositives(name='fp'),
            tf2.keras.metrics.TrueNegatives(name='tn'),
            tf2.keras.metrics.FalseNegatives(name='fn'),
            tf2.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf2.keras.metrics.Precision(name='precision'),
            tf2.keras.metrics.Recall(name='recall'),
            tf2.keras.metrics.AUC(name='auc'),
            tf2.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ]
    else:
        metrics = [
            tf2.keras.metrics.MeanSquaredError(name='MSE'),
            tf2.keras.metrics.MeanAbsoluteError(name='MAE'),
            tf2.keras.metrics.RootMeanSquaredError(name='RMSE'),
        ]

    # TODO why are we doing categorical and not binary_crossentropy?
    loss = 'categorical_crossentropy' if goal == 'classification' else 'mean_absolute_error'
    model.compile(loss=loss, optimizer='Adam', metrics=metrics)
    model.summary()
    GLOBAL_KERAS_MODEL = model

    ## Defining NN architecture
    l_pred, units, clean_prob = fc_network(variables, tf_X, n_layers, activ_f = activ_f)
    l_adv_fair, units_adv, adv_prob = fc_network(variables, tf_adv_X, n_layers, activ_f = activ_f)

    if goal == 'classification':
        cross_entropy = tf1.reduce_mean(
            tf1.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=l_pred))
    elif goal == 'regression':
        cross_entropy = tf1.reduce_mean(
            tf1.losses.mean_squared_error(labels=tf_y, predictions=clean_prob))
    else:
        raise ValueError(f'Invalid training goal {goal}')

    print(f'\ntf_y = {tf_y.shape}')
    print(f'\nl_pred_fair = {l_adv_fair.shape}')

    # Creating the fair loss for MILP
    a_lam = tf1.math.subtract(1.0, lam)
    if goal == 'classification':
        cross_entropy_pt1 = tf1.math.multiply(
            lam, tf1.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=l_pred))
    elif goal == 'regression':
        cross_entropy_pt1 = tf1.math.multiply(
            lam, tf1.losses.mean_squared_error(labels=tf_y, predictions=clean_prob))
    else:
        raise ValueError(f'Invalid training goal {goal}')
    mean_abs_err_pt2 = tf1.math.multiply(
        a_lam, tf1.losses.absolute_difference(labels=clean_prob, predictions=adv_prob))
    cross_entropy_fair = tf1.reduce_mean(tf1.math.add(cross_entropy_pt1, mean_abs_err_pt2))

    correct_prediction = tf1.equal( tf1.math.greater(l_pred,0) , tf1.math.greater(tf_y,0))
    accuracy = tf1.reduce_mean(tf1.cast(correct_prediction, tf1.float32))

    if l2_reg > 0:
        cross_entropy += l2_reg*sum(
            [tf1.nn.l2_loss(variables['weight_' + str(l)]) for l in range(n_layers)])
        cross_entropy_fair += l2_reg*sum(
            [tf1.nn.l2_loss(variables['weight_' + str(l)]) for l in range(n_layers)])

    return variables, l_pred, cross_entropy, accuracy, cross_entropy_fair, GLOBAL_KERAS_MODEL


def train_fair_nn_binary(X_train, y_train, sensitive_directions, X_test=None, y_test=None, config=None, train_ds=None, save_directory=None):

    start_time = time()

    goal = config['training_goal']
    epoch = config['fair_epochs']
    l2_reg = config['reg']
    n_units = config['n_units']
    lr = config['lr']
    batch_size = config['fair_batch_size']
    verbose=True
    subspace_epoch=10
    subspace_step=.1
    eps=None
    full_step=-1
    full_epoch=10
    fair_start = True
    activ_f = tf1.nn.relu
    lamb_init=2.
    weights=None

    use_MILP = config['training_MILP']
    epsilon = config['epsilon']
    delta = config['delta']
    opt_mode = config['training_opt_mode']
    time_limit = config['training_verif_time_limit']
    lam = config['lambda']

    if fair_start:
        fair_start = epoch/2
    else:
        fair_start = 0

    ## Fair distance
    proj_compl = compl_svd_projector(sensitive_directions, svd=-1)
    dist_f = fair_dist(proj_compl, 0.)
    V_sensitive = sensitive_directions.shape[0]

    global_step = tf1.compat.v1.train.get_or_create_global_step()

    lamb = lamb_init
    N, D = X_train.shape
    K = y_train.shape[1]

    n_per_class = int(batch_size/2)
    num_ones = y_train.sum(axis=0)[0]
    num_zeros = y_train.shape[0] - num_ones
    n_per_class = int(min(n_per_class, min(num_ones, num_zeros)))

    tf_X = tf1.placeholder(tf1.float32, shape=[None,D], name='clean_inputs')
    tf_y = tf1.placeholder(tf1.float32, shape=[None,K], name='response')

    # MILP Adversarial inputs
    tf_adv_X = tf1.placeholder(tf1.float32, shape=[None,D], name='adv_inputs')

    ## Fair variables
    tf_directions = tf1.constant(sensitive_directions, dtype=tf1.float32)

    variables, l_pred, _, accuracy, loss, GLOBAL_KERAS_MODEL = forward_fair_binary(
        tf_X, tf_y, tf_adv_X, lam, n_units = n_units, activ_f = activ_f, l2_reg=l2_reg, goal=goal)

    optimizer = tf1.train.AdamOptimizer(learning_rate=lr)
    train_step = optimizer.minimize(
        loss, var_list=list(variables.values()), global_step=global_step)
    reset_optimizer = tf1.variables_initializer(optimizer.variables())
    reset_main_step = True

    if full_step > 0:
        full_fair_optimizer = tf1.train.AdamOptimizer(learning_rate=full_step)
        full_fair_step = full_fair_optimizer.minimize(
            loss, var_list=[full_adv_weights], global_step=global_step)
        reset_full_fair_optimizer = tf1.variables_initializer(full_fair_optimizer.variables())
        reset_full_adv_weights = full_adv_weights.assign(tf1.zeros([batch_size,D]))

    failed_attack_count = 0
    failed_full_attack = 0
    failed_subspace_attack = 0

    out_freq = 10

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        for it in range(epoch):

            if goal == 'classification':
                batch_idx = sample_batch_idx_binary(y_train, n_per_class)
            else:
                batch_idx = sample_batch_idx_regression(y_train, batch_size)
            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]

            if it > fair_start:
                if reset_main_step:
                    sess.run(reset_optimizer)
                    reset_main_step = False


                # Here we pass adv_x as just x as it results in 0 MSE and returns
                # the normal loss
                loss_before_subspace_attack = loss.eval(feed_dict={
                            tf_X: batch_x, tf_y: batch_y, tf_adv_X: batch_x})

                ## Do subspace attack
                if(use_MILP is False):
                    print("NOPE NOT IMPLIMENTED")
                else:
                    model_weights = []
                    global_weights = []
                    for key in variables:
                        var = np.asarray(variables[key].eval())
                        model_weights.append(var)
                        global_weights.append(var.tolist())
                    nproc = GLOBAL_NPROC
                    p = Pool(nproc)
                    args = []
                    sub_batch_size = int(batch_size/nproc)
                    GLOBAL_KERAS_MODEL.set_weights(model_weights)
                    for i in range(nproc):
                        # having nproc copies of train_ds makes me nervous about RAM usage bottlenecks, but it works for now
                        subbatch_idx = list(batch_idx[sub_batch_size*i:sub_batch_size*(i+1)])
                        verification_proc_id = uuid.uuid4()
                        arg = (train_ds, subbatch_idx,  proj_compl, epsilon, delta, opt_mode, time_limit, False, global_weights, None, verification_proc_id)
                        args.append((arg))
                    batch_x_MILP = p.map(multiproc_veri, args)
                    p.close()
                    p.join()
                    batch_x_MILP = np.asarray(batch_x_MILP)
                    batch_x_MILP = np.concatenate(batch_x_MILP, axis=0)

                    print("INFO BATCH RESULT: ", type(batch_x_MILP), batch_x_MILP.shape)

                    loss_after_subspace_attack = loss.eval(feed_dict={
                                tf_X: batch_x, tf_y: batch_y, tf_adv_X: batch_x_MILP})

                if full_step > 0:

                    ## Do full attack
                    if(use_MILP is False):
                        print("NOT IMPLEMENTED")
                    else:
                        model_weights = []
                        global_weights = []
                        for key in variables:
                            var = np.asarray(variables[key].eval())
                            model_weights.append(var)
                            global_weights.append(var.tolist())
                        nproc = GLOBAL_NPROC
                        p = Pool(nproc)
                        args = []
                        sub_batch_size = int(batch_size/nproc)
                        GLOBAL_KERAS_MODEL.set_weights(model_weights)
                        for i in range(nproc):
                            # having nproc copies of train_ds makes me nervous about RAM usage bottlenecks, but it works for now
                            subbatch_idx = list(batch_idx[sub_batch_size*i:sub_batch_size*(i+1)])
                            verification_proc_id = uuid.uuid4()
                            arg = (train_ds, subbatch_idx,  proj_compl, epsilon, delta, opt_mode, time_limit, False, global_weights, batch_x, verification_proc_id)
                            args.append((arg))
                        batch_x_MILP = p.map(multiproc_veri, args)
                        p.close()
                        p.join()
                        batch_x_MILP = np.asarray(batch_x_MILP)
                        batch_x_MILP = np.concatenate(batch_x_MILP, axis=0)
                        print("INFO BATCH RESULT: ", type(batch_x_MILP), batch_x_MILP.shape)
                        loss_after_l2_attack = loss.eval(feed_dict={
                                tf_X: batch_x_MILP, tf_y: batch_y, tf_lamb: lamb, tf_adv_X: batch_x_MILP})

                if(use_MILP is False):
                    adv_batch = tf_fair_X.eval(feed_dict={tf_X: batch_x})
                else:
                    adv_batch = batch_x_MILP

                if eps is not None:
                    # Here is their weird distributional loss thing that you were asking about
                    mean_dist = dist_loss.eval(feed_dict={tf_X: batch_x})
                    lamb = max(
                        0.00001,lamb + (max(mean_dist,eps)/min(mean_dist,eps))*(mean_dist - eps))
            else:
                adv_batch = batch_x
                batch_x_MILP = batch_x
            _, loss_at_update = sess.run([train_step,loss], feed_dict={
                  tf_X: batch_x, tf_y: batch_y, tf_adv_X: batch_x_MILP})

#            if it % out_freq == 0 and verbose:
            if True:
                train_acc, train_logits = sess.run([accuracy,l_pred], feed_dict={
                      tf_X: X_train, tf_y: y_train})
                print('Epoch %d train accuracy %f; lambda is %f' % (it, train_acc, lamb))
                if y_test is not None:
                    test_acc, test_logits = sess.run([accuracy,l_pred], feed_dict={
                            tf_X: X_test, tf_y: y_test})
                    print('Epoch %d test accuracy %g' % (it, test_acc))

                ## Attack summary
                if it > fair_start:
                    print('FAILED attacks: subspace %d; full %d; Nans after attack %d' % (failed_subspace_attack, failed_full_attack, failed_attack_count))
                    print('Loss clean %f; subspace %f; full %f' % (loss_before_subspace_attack, loss_after_subspace_attack, loss_at_update))

        if y_train is not None:
            print('\nFinal train accuracy %g' % (accuracy.eval(feed_dict={
                  tf_X: X_train, tf_y: y_train})))
        if y_test is not None:
            print('Final test accuracy %g' % (accuracy.eval(feed_dict={
                  tf_X: X_test, tf_y: y_test})))
        if eps is not None:
            print('Final lambda %f' % lamb)

        fair_weights = [x.eval() for x in variables.values()]
        train_logits = l_pred.eval(feed_dict={tf_X: X_train})
        if X_test is not None:
            test_logits = l_pred.eval(feed_dict={tf_X: X_test})
        else:
            test_logits = None

        model_weights = []
        for key in variables:
            var = np.asarray(variables[key].eval())
            model_weights.append(var)
        GLOBAL_KERAS_MODEL.set_weights(model_weights)

        end_time = time()
        training_time = end_time - start_time
        config['training_time'] = training_time

        models_dir = save_directory or 'saved_models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        model_id = f'{models_dir}/{datetime.now()}---{uuid.uuid4()}'
        os.makedirs(model_id)

        with open(f'{model_id}/info.json', 'w') as f:
            import json
            json.dump(config, f)

        tf2.keras.models.save_model(GLOBAL_KERAS_MODEL, f'{model_id}/model')
    return fair_weights, train_logits, test_logits, GLOBAL_KERAS_MODEL

