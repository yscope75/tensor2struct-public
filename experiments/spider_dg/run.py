#!/usr/bin/env python

import os
import _jsonnet
import json
import argparse
import attr
import wandb
import optuna

from experiments.spider_dg import (
    train,
    meta_train,
    dema_train,
    bayesian_meta_train,
    eqrm_train,
)


# global variables
exp_config = dict()
model_config_file = ''
model_config_args = ''


@attr.s
class TrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


@attr.s
class MetaTrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


@attr.s
class DEMATrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


@attr.s
class EQRMTrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    

def train(type):
    if type == "train":
        train_config = TrainConfig(model_config_file, model_config_args, logdir)
        train.main(train_config)
    elif type == "meta_train":
        train_config = MetaTrainConfig(model_config_file, model_config_args, logdir)
        meta_train.main(train_config)
    elif type == "dema_train":
        train_config = DEMATrainConfig(model_config_file, model_config_args, logdir)
        dema_train.main(train_config)
    elif type == "bayesian_meta_train":
        train_config = MetaTrainConfig(model_config_file, model_config_args, logdir)
        bayesian_meta_train.main(train_config)
    elif type == "eqrm_train":
        train_config = EQRMTrainConfig(model_config_file, model_config_args, logdir)
        return eqrm_train.main(train_config)  # return score on dev set


# define an objective function to be maximized
def objective(trial):
    # suggest values of the hyperparameters for eqrm
    exp_config['model_config_args']['bs'] = trial.suggest_int("BATCHSIZE", 8, 24, 8)
    exp_config['model_config_args']['lr'] = trial.suggest_float("LR", 1e-6, 6e-4, log=True)
    exp_config['model_config_args']['quantile'] = trial.suggest_float("QUANTILE", 0.2, 1, step=0.05) 
    exp_config['model_config_args']['burnin_iters'] = trial.suggest_int("BURNIN_ITERS", 0, 10000, step=500)
    exp_config['model_config_args']['num_warmup_steps'] = trial.suggest_int("NUM_WARMUP_STEPS", 500, 2000, step=500)
        
    # reload model config args
    if "model_config_args" in exp_config:
        model_config_args = json.dumps(exp_config["model_config_args"])
    else:
        model_config_args = None  
    
    # score on dev set
    return train("eqrm_train")


def main():
    global exp_config, model_config_file, model_config_args, logdir
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "mode", choices=["train", "meta_train", "dema_train", "bayesian_meta_train", "eqrm_train", "params_search"], help="train/meta_train/dist_train",
    # )
    # parser.add_argument("exp_config_file", help="jsonnet file for experiments")
    args = parser.parse_args()
    args.mode = "params_search"
    # args.exp_config_file = '/media/doublemint/SharedDisk/repo/SummerProject/Text-to-SQL/tensor2struct-yscope75/tensor2struct-public/configs/spider/run_config/run_spider_dgmaml_b.jsonnet'
    args.exp_config_file = '/media/doublemint/SharedDisk/repo/SummerProject/Text-to-SQL/tensor2struct-yscope75/tensor2struct-public/configs/spider/run_config/run_spider_eqrm.jsonnet'
    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = json.dumps(exp_config["model_config_args"])
    else:
        model_config_args = None
    
    # cluster base dir
    log_base_dir = os.environ.get("LOG_BASE_DIR", None)
    if log_base_dir is None:
        print(f"Using default log base dir {os.getcwd()}")
        logdir = exp_config["logdir"]
    else:
        logdir = os.path.join(log_base_dir, exp_config["logdir"])

    # wandb init
    expname = exp_config["logdir"].split("/")[-1]
    project = exp_config["project"]
    
    # dist train need to start a wandb session in each process, not a global one
    if args.mode in ["train", "meta_train", "dema_train", "bayesian_meta_train", "eqrm_train", "params_search"]:
        wandb.init(project=project, group=expname, job_type=args.mode)
    
    if args.mode == "params_search":
        # create a study object and optimize the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
    else:
        train(args.mode)


if __name__ == "__main__":
    main()
