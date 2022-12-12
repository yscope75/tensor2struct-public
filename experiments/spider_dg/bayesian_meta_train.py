import argparse
import collections
import copy
import datetime
import json
import os
import logging
import itertools

import _jsonnet
import attr
import numpy as np
import torch
import torch.nn as nn
import wandb

from tensor2struct.utils import registry, random_state, vocab
from tensor2struct.utils import saver as saver_mod
from tensor2struct.commands import train, meta_train
from tensor2struct.training import maml


@attr.s
class MetaTrainConfig(meta_train.MetaTrainConfig):
    use_bert_training = attr.ib(kw_only=True)

class BayesianMLDG(nn.Module):
    def __init__(
        self,
        base_model,
        config
    ):
        # todo: refer rat config 
        super().__init__()
        assert config.num_particles >= 2, ValueError("At least 2 particle")
        self.config = config 
        self.base_model_dict = base_model.state_dict()
        self.parameter_shapes = []
        # not include SVGD for bert
        for key in self.base_model_dict:
            self.parameter_shapes.append(self.base_model_dict[key].shape)
            # to do: separate bert params with non-bert params here
            
        self.params = nn.ParameterList(parameters=None)
        
        # init params for each model
        for _ in range(self.config.num_particles):
            params_list = self.init_params(state_dict=self.base_model_dict)
            params_vector = nn.utils.parameters_to_vector(parameters=params_list) 
            self.params.append(parameter=nn.Parameter(data=params_vector))
            
        # num of params used for computing distance
        self.num_base_params = self.params[0].size(0)
    
    def init_params(self, state_dict):
        params = [state_dict[key] for key in state_dict if 'bert' not in key]
        params = []
        for key in state_dict:
            if 'bert' not in key:
                param = state_dict[key]
                if param.ndim > 1:
                    torch.nn.init.xavier_normal_(param)
                else:
                    torch.nn.init.zeros_(param)
                params.append(param)
            else:
                param = copy.deepcopy(param)

        return params 
    def vector_to_list_params(self, params_vec):
        params = []
        # pointer for each layer params
        pointer = 0 
        
        for param_shape in self.params_shapes:
            # total number of params each layer
            num_params = int(np.prod(param_shape))
            
            params.append(params_vec[pointer:pointer+num_params].view(param_shape))
            
            pointer += num_params

        return params
    
    def forward(self, particle_id):
        
        particle_params = self.vector_to_list_params(self.params[particle_id])
        return particle_params

class BayesianMetaTrainer(meta_train.MetaTrainer):
    def load_train_config(self):
        self.train_config = registry.instantiate(
            MetaTrainConfig, self.config["meta_train"]
        )

        if self.train_config.num_batch_accumulated > 1:
            self.logger.warn("Batch accumulation is used only at MAML-step level")

        if self.train_config.use_bert_training:
            if self.train_config.clip_grad is None:
                self.logger.info("Gradient clipping is recommended for BERT training")

    def load_optimizer(self, config):
        with self.init_random:
            # 0. create inner_optimizer
            # inner_parameters = list(self.model.get_trainable_parameters())
            inner_parameters = list(self.model.get_non_bert_parameters())
            inner_optimizer = registry.construct(
                "optimizer", self.train_config.inner_opt, params=inner_parameters
            )
            self.logger.info(f"{len(inner_parameters)} parameters for inner update")

            # 1. MAML trainer, might add new parameters to the optimizer, e.g., step size
            maml_trainer = maml.MAML(
                model=self.model, inner_opt=inner_optimizer, device=self.device,
            )
            maml_trainer.to(self.device)

            opt_params = maml_trainer.get_inner_opt_params()
            self.logger.info(f"{len(opt_params)} opt meta parameters")

            # 2. Outer optimizer
            # if config["optimizer"].get("name", None) in ["bertAdamw", "torchAdamw"]:
            if self.train_config.use_bert_training:
                bert_params = self.model.get_bert_parameters()
                non_bert_params = self.model.get_non_bert_parameters()
                assert len(non_bert_params) + len(bert_params) == len(
                    list(self.model.parameters())
                )
                assert len(bert_params) > 0
                self.logger.info(
                    f"{len(bert_params)} BERT parameters and {len(non_bert_params)} non-BERT parameters"
                )

                optimizer = registry.construct(
                    "optimizer",
                    config["optimizer"],
                    non_bert_params=non_bert_params,
                    bert_params=bert_params,
                )
                lr_scheduler = registry.construct(
                    "lr_scheduler",
                    config.get("lr_scheduler", {"name": "noop"}),
                    param_groups=[
                        optimizer.non_bert_param_group,
                        optimizer.bert_param_group,
                    ],
                )
            else:
                optimizer = registry.construct(
                    "optimizer",
                    config["optimizer"],
                    params=self.model.get_trainable_parameters(),
                )
                lr_scheduler = registry.construct(
                    "lr_scheduler",
                    config.get("lr_scheduler", {"name": "noop"}),
                    param_groups=optimizer.param_groups,
                )

            lr_scheduler = registry.construct(
                "lr_scheduler",
                config.get("lr_scheduler", {"name": "noop"}),
                param_groups=optimizer.param_groups,
            )
            return inner_optimizer, maml_trainer, optimizer, lr_scheduler

    def step(
        self,
        config,
        train_data_scheduler,
        maml_trainer,
        optimizer,
        lr_scheduler,
        last_step,
    ):
        with self.model_random:
            for _i in range(self.train_config.num_batch_accumulated):
                task = train_data_scheduler.get_batch(last_step)
                inner_batch, outer_batches = task
                ret_dic = maml_trainer.meta_train(
                    self.model, inner_batch, outer_batches
                )
                loss = ret_dic["loss"]

            # clip bert grad
            if self.train_config.clip_grad and self.train_config.use_bert_training:
                for param_group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(
                        param_group["params"], self.train_config.clip_grad,
                    )


            optimizer.step()
            optimizer.zero_grad()

            # log lr for each step
            outer_lr = lr_scheduler.update_lr(last_step)
            if outer_lr is None:
                outer_lr = [param["lr"] for param in optimizer.param_groups]
            inner_lr = [param["lr"] for param in maml_trainer.inner_opt.param_groups]

        # Report metrics and lr
        if last_step % self.train_config.report_every_n == 0:
            self.logger.info("Step {}: loss={:.4f}".format(last_step, loss))
            self.logger.info(f"Step {last_step}, lr={inner_lr, outer_lr}")
            wandb.log({"train_loss": loss}, step=last_step)
            for idx, lr in enumerate(inner_lr):
                wandb.log({f"inner_lr_{idx}": lr}, step=last_step)
            for idx, lr in enumerate(outer_lr):
                wandb.log({f"outer_lr_{idx}": lr}, step=last_step)


def main(args):
    # setup logger etc
    config, logger = train.setup(args)

    # Construct trainer and do training
    trainer = MetaTrainer(logger, config)
    trainer.train(config, modeldir=args.logdir)


if __name__ == "__main__":
    args = train.add_parser()
    main(args)
