from time import time
import numpy as np

from dataset.utils import (
    get_adult_data,
    get_credit_data,
    PROJ_ADULT_FILENAME,
    SSIF_WEIGHTS_ADULT_FILENAME,
    VANILLA_WEIGHTS_ADULT_FILENAME,
    PROJ_CREDIT_FILENAME,
    SSIF_WEIGHTS_CREDIT_FILENAME,
    VANILLA_WEIGHTS_CREDIT_FILENAME,
)

from verification.global_v import verify_globally_for_mahalanobis, verify_globally_for_weighted_lp
from verification.local_v import verify_locally

# TODO: need to think about random seed
def verify(
    train_ds,
    test_ds,
    model,
    proj,
    config,
    proc_id,
):

    strategy = config['strategy']
    opt_mode = config['opt_mode']
    time_limit = config['time_limit']
    epsilon = config.get('verification_epsilon') or config['epsilon']
    delta = config['delta']
    M = config['M']
    mip = config['mip']
    dist_metric = config['dist_metric']

    if strategy == 'local':
        verification_results = verify_locally(
            test_ds, model, proj=proj, epsilon=epsilon,
            delta=delta, opt_mode=opt_mode, time_limit=time_limit)
    elif strategy == 'global':
        if dist_metric == 'mahalanobis':
            verification_results = verify_globally_for_mahalanobis(
                test_ds, model, proj=proj, epsilon=epsilon, delta=delta, opt_mode=opt_mode, M=M,
                time_limit=time_limit, mip=mip, proc_id=proc_id)
        elif dist_metric == 'weighted_lp':
            verification_results = verify_globally_for_weighted_lp(
                test_ds, model, epsilon=epsilon, delta=delta, opt_mode=opt_mode, M=M,
                time_limit=time_limit, mip=mip, proc_id=proc_id)
        else:
            raise ValueError(f'Unrecognised distance metric {dist_metric}')
    else:
        raise ValueError(f'{strategy} is not a valid verification strategy name')

    return verification_results

if __name__ == "__main__":
    epsilon = .01
    delta = 0.1

    sensitive_features = ['sex']
    drop_columns = ['native-country', 'education']
    train_ds, test_ds = get_adult_data(sensitive_features, drop_columns=drop_columns)
    proj_filename = PROJ_ADULT_FILENAME
    # weights_filename = SSIF_WEIGHTS_ADULT_FILENAME
    weights_filename = VANILLA_WEIGHTS_ADULT_FILENAME


    verify(
        train_ds,
        test_ds,
        strategy='global',
        opt_mode='milp',
        epsilon=epsilon,
        delta=delta,
        proj_filename=proj_filename,
        weights_filename=weights_filename
    )


# aggiungere aprossimazione con sampling

