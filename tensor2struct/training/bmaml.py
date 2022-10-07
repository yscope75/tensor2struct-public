import torch 
import torch.nn as nn 
import numpy as np
import higher 
import logging

class BayesianMAML(nn.Module):
    def __init__(
        self,
        base_model,
        config
    ):
        # todo: refer rat config 
        super().__init__()
        assert config.num_particles >= 1, ValueError("At least 1 particle")
        self.config = config 
        self.base_model = base_model
        self.base_model_dict = base_model.state_dict()
        self.parameter_shapes = []
        
        for param in self.base_model_dict.values():
            self.parameter_shapes.append(param.shape)
            
        self.params = nn.ParameterList(parameters=None)
        
        # init params for each model
        for _ in range(self.config.num_particles):
            params_list = init_params(state_dict=self.base_model_dict)
            params_vector = nn.utils.parameters_to_vector(parameters=params_list) 
            self.params.append(parameter=nn.Parameter(data=params_vector))
            
        # num of params used for computing distance
        self.num_base_params = np.sum([torch.numel(p) for p in self.params[0]])
            
    def forward(self, particle_id):
       return vector_to_list_params(
           self.params[particle_id], 
           param_shapes=self.parameter_shapes)
            
class BayesianDGAgnosticMetaLearning(nn.Module):
    def __init__(
        self, 
        model=None, 
        inner_opt=None,
        first_order=False,
        device=None,
        config=None
    ):
        super().__init__()
        self.inner_opt = inner_opt
        self.first_order = first_order
        self.inner_steps = 1
        self.device = device
        self.config = config
        
        self.f_model = self.convert_to_stateless(model)
        # create particles of model
        self.bdgmaml = BayesianMAML(model, self.config)
        
    def bdgmaml_train(self, model, inner_batch, outer_batch):
        assert model.trainning
        ret_dict = {}
        
        # create stateless essemble model
        self.f_essemble_model = higher.patch.monkeypatch(
            module=self.bmlma,
            copy_initial_weights=False,
            track_higher_grads=self.flags.track_higher_grads
        )
        bdgmaml_params = torch.stack(tensors=[p for p in self.bmlma.parameters()])
        
        for _step in range(self.inner_steps):
            
            # create matrix for computing distance
            distance_NLL = torch.empty(
                size=(self.flags.num_particles, 
                      self.bmlma.num_base_params), 
                device=self.device)
            
        
        
        
    @staticmethod
    def convert_to_stateless(model):
        
        f_model = higher.patch.make_functional(module=model)
        f_model.track_higher_grads = False
        f_model._fast_params = [[]]

        return f_model