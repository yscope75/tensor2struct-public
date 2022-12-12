import copy
import torch 
import higher
import torch.nn as nn 
import numpy as np
import logging

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
            if 'bert' not in key:
                self.parameter_shapes.append(self.base_model_dict[key].shape)
            
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
            
class BayesianDGAgnosticMetaLearning(nn.Module):
    def __init__(
        self, 
        inner_opt=None,
        device=None,
        f_model=None,
        config=None
    ):
        super().__init__()
        self.inner_opt = inner_opt
        self.inner_steps = 1
        self.device = device
        self.config = config
        self.f_model = f_model
        
    def bdgmaml_train(self, hyper_model, inner_batch, outer_batch):
        assert hyper_model.trainning
        ret_dict = {}
        
        # create stateless essemble model
        f_ensemble_model = higher.patch.monkeypatch(
            module=hyper_model,
            copy_initial_weights=False,
            track_higher_grads=self.config.track_higher_grads
        )
        # use for updating ensemble model later
        ensemble_params = torch.stack(tensors=[p for p in hyper_model.parameters()])
        
        ret_dict = {}
        for _step in range(self.inner_steps):
            
            # create matrix for computing distance
            distance_NLL = torch.empty(
                size=(self.config.num_particles, 
                      hyper_model.num_base_params), 
                device=self.device)
            for particle_id in range(self.config.num_particles):
                particle_params = f_ensemble_model.forward(particle_id=particle_id)
                particle_ret_dict = self.f_model(inner_batch, params=particle_params)
                particle_loss = particle_ret_dict["loss"]
                
                if self.config.first_order:
                    grads = torch.autograd.grad(
                        outputs=particle_loss,
                        inputs=f_ensemble_model.fast_params[particle_id],
                        retain_graph=True
                    )
                else:
                    grads = torch.autograd.grad(
                        output=particle_loss,
                        inputs=f_ensemble_model.fast_params[particle_id],
                        create_graph=True
                    )
                
                distance_NLL[particle_id, :] = nn.utils.parameters_to_vector(parameters=grads)

            # compute kernel distances and grads
            kernel_matrix, grad_kernel, _ = self.get_kernel(params=ensemble_params)
            
            q_params = q_params - self.config.inner_lr * (torch.matmul(kernel_matrix, distance_NLL) - grad_kernel)
            
            f_ensemble_model.update_params(params)
        
        
    def get_kernel(self, params: torch.Tensor):
        """
        Compute the RBF kernel for the input
        
        Args:
            params: a tensor of shape (N, M)
        
        Returns: kernel_matrix = tensor of shape (N, N)
        """
        pairwise_d_matrix = self.get_pairwise_distance_matrix(x=params)

        median_dist = torch.quantile(input=pairwise_d_matrix, q=0.5)  # tf.reduce_mean(euclidean_dists) ** 2
        h = median_dist / np.log(self.config.num_particles)

        kernel_matrix = torch.exp(-pairwise_d_matrix / h)
        kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
        grad_kernel = -torch.matmul(kernel_matrix, params)
        grad_kernel += params * kernel_sum
        grad_kernel /= h

        return kernel_matrix, grad_kernel, h

    @staticmethod
    def get_pairwise_distance_matrix(x: torch.Tensor) -> torch.Tensor:
        """Calculate the pairwise distance between each row of tensor x
        
        Args:
            x: input tensor
        
        Return: matrix of point-wise distances
        """
        n, m = x.shape

        # initialize matrix of pairwise distances as a N x N matrix
        pairwise_d_matrix = torch.zeros(size=(n, n), device=x.device)

        # num_particles = particle_tensor.shape[0]
        euclidean_dists = torch.nn.functional.pdist(input=x, p=2) # shape of (N)

        # assign upper-triangle part
        triu_indices = torch.triu_indices(row=n, col=n, offset=1)
        pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

        # assign lower-triangle part
        pairwise_d_matrix = torch.transpose(pairwise_d_matrix, dim0=0, dim1=1)
        pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

        return pairwise_d_matrix