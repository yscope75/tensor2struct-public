import attr
import math
import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd 
import collections
import copy
import logging

logger = logging.getLogger("tensor2struct")

# Todo: move this class for universal use
@attr.s
class SpiderEncoderState:
    state = attr.ib()
    memory = attr.ib()
    question_memory = attr.ib()
    schema_memory = attr.ib()
    words_for_copying = attr.ib()

    pointer_memories = attr.ib()
    pointer_maps = attr.ib()

    m2c_align_mat = attr.ib()
    m2t_align_mat = attr.ib()

    # for copying
    tokenizer = attr.ib()

    def find_word_occurrences(self, token):
        occurrences = [i for i, w in enumerate(self.words_for_copying) if w == token]
        if len(occurrences) > 0:
            return occurrences[0]
        else:
            return None

class BayesModelAgnosticMetaLearning(nn.Module):
    def __init__(
        self, 
        inner_lr,
        model=None, 
        first_order=False, 
        device=None, 
        num_particles=2,
    ):
        super().__init__()
        self.inner_lr = inner_lr
        self.first_order = first_order
        self.inner_steps = 1
        self.num_particles = num_particles
        self.device = device

    def get_inner_opt_params(self):
        """
        Equvalent to self.parameters()
        """
        return []

    def meta_train(self, model, 
                   model_encoder_params,
                   model_aligner_params,
                   model_decoder_params, 
                   inner_batch, outer_batches):
        assert not self.first_order
        return self.maml_train(model, 
                               model_encoder_params, 
                               model_aligner_params,
                               model_decoder_params, 
                               inner_batch, 
                               outer_batches)

    def maml_train(self, 
                   model, 
                   model_encoder_params,
                   model_aligner_params,
                   model_decoder_params,
                   inner_batch, 
                   outer_batches):
        assert model.training
        # clone model for inner gradients computing
        inner_encoders = copy.deepcopy(model.list_of_encoders)
        inner_aligner = copy.deepcopy(model.aligner)
        inner_decoder = copy.deepcopy(model.decoder)
        inner_encoder_params = []
        for i in range(self.num_particles):
            inner_encoder_params.append(list(inner_encoders[i].parameters()))
        inner_params_matrix = torch.stack(
            [torch.nn.utils.parameters_to_vector(params) for params in inner_encoder_params],
            dim=0
        )
        model_bert_params = list(model.bert_model.parameters())
        bert_model_len = len(model_bert_params)
        inner_aligner_params = list(inner_aligner.parameters())
        inner_decoder_params = list(inner_decoder.parameters())
        particle_len = len(inner_encoder_params[0])
        aligner_len = len(inner_aligner_params)
        inner_aligner_p_vec = torch.nn.utils.parameters_to_vector(inner_aligner_params)
        inner_decoder_p_vec = torch.nn.utils.parameters_to_vector(inner_decoder_params)
        ret_dic = {}
        for _step in range(self.inner_steps):
            # for computing distance 
            distance_nll = torch.empty(size=(self.num_particles,
                                             inner_params_matrix.size(1)),
                                       device=self.device)
            # decoder grad vector, store decoder grads on inner loop
            alinger_grads_vec = torch.zeros_like(inner_aligner_p_vec)
            decoder_grads_vec = torch.zeros_like(inner_decoder_p_vec)
            enc_input_list = [enc_input for enc_input, dec_output in inner_batch]
            column_pointer_maps = [
                {i: [i] for i in range(len(desc["columns"]))} for desc in enc_input_list
            ]
            table_pointer_maps = [
                {i: [i] for i in range(len(desc["tables"]))} for desc in enc_input_list
            ]
            inner_loss = []
            # for single input source domain
            with torch.no_grad():
                plm_output = model.bert_model(enc_input_list)
            for i in range(self.num_particles):
                
                enc_states = []
                for idx, (enc_input, plm_out) in enumerate(zip(enc_input_list, plm_output)):
                    relation = model.schema_linking(enc_input)
                    (
                        q_enc_new_item,
                        c_enc_new_item,
                        t_enc_new_item,
                    ) = inner_encoders[i](enc_input, 
                                          plm_out,
                                          relation)
                    # attention memory 
                    memory = []
                    include_in_memory = inner_encoders[0].include_in_memory
                    if "question" in include_in_memory:
                        memory.append(q_enc_new_item)
                    if "column" in include_in_memory:
                        memory.append(c_enc_new_item)
                    if "table" in include_in_memory:
                        memory.append(t_enc_new_item)
                    memory = torch.cat(memory, dim=1)
                    # alignment matrix
                    align_mat_item = inner_aligner(
                        enc_input, q_enc_new_item, c_enc_new_item, t_enc_new_item, relation
                    )
                    enc_states.append(
                        SpiderEncoderState(
                            state=None,
                            words_for_copying=enc_input["question_for_copying"],
                            tokenizer=inner_encoders[0].tokenizer,
                            memory=memory,
                            question_memory=q_enc_new_item,
                            schema_memory=torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                            pointer_memories={
                                "column": c_enc_new_item,
                                "table": t_enc_new_item,
                            },
                            pointer_maps={
                                "column": column_pointer_maps[idx],
                                "table": table_pointer_maps[idx],
                            },
                            m2c_align_mat=align_mat_item[0],
                            m2t_align_mat=align_mat_item[1],
                        )
                    )

                losses = []
                for enc_state, (enc_input, dec_output) in zip(enc_states, inner_batch):
                    ret_dic = inner_decoder(dec_output, enc_state)
                    losses.append(ret_dic["loss"])
                loss = torch.mean(torch.stack(losses, dim=0), dim=0)
                inner_loss.append(loss.item())
                enc_dec_grads = torch.autograd.grad(loss, 
                                                    inner_encoder_params[i] + inner_aligner_params + inner_decoder_params,
                                                    allow_unused=True)
                particle_grads = enc_dec_grads[:particle_len]
                aligner_grads = list(enc_dec_grads[particle_len:particle_len + aligner_len])
                for idx, g in enumerate(aligner_grads):
                    if g is None:
                        aligner_grads[idx] = torch.zeros_like(inner_aligner_params[idx])
                decoder_grads = list(enc_dec_grads[particle_len + aligner_len:])
                for idx, g in enumerate(decoder_grads):
                    if g is None:
                        decoder_grads[idx] = torch.zeros_like(inner_decoder_params[idx])
                        
                alinger_grads_vec = alinger_grads_vec + (1/self.num_particles)*torch.nn.utils.parameters_to_vector(aligner_grads)
                decoder_grads_vec = decoder_grads_vec + (1/self.num_particles)*torch.nn.utils.parameters_to_vector(decoder_grads)
                
                distance_nll[i, :] = torch.nn.utils.parameters_to_vector(particle_grads)
                
            # del particle_grads
            # del aligner_grads
            # del decoder_grads
            # del enc_dec_grads
            # gc.collect()
            
            grad_kernel, _ = BayesModelAgnosticMetaLearning.get_kernel_wSGLD_B(params=inner_params_matrix,
                                              num_of_particles=self.num_particles)
            
            # compute inner gradients with rbf kernel
            inner_grads = distance_nll - grad_kernel
            # update inner_net parameters 
            inner_params_matrix = inner_params_matrix - self.inner_lr*inner_grads
            for i in range(self.num_particles):
                torch.nn.utils.vector_to_parameters(inner_params_matrix[i],
                                                    inner_encoder_params[i])
            del inner_params_matrix
            # update aligner parameters
            inner_aligner_p_vec = inner_aligner_p_vec - self.inner_lr*alinger_grads_vec
            torch.nn.utils.vector_to_parameters(inner_aligner_p_vec, inner_aligner_params)
            # update decoder parameters
            inner_decoder_p_vec = inner_decoder_p_vec - self.inner_lr*decoder_grads_vec
            torch.nn.utils.vector_to_parameters(inner_decoder_p_vec, inner_decoder_params)
            # clear var for saving memory
            # del inner_aligner_p_vec
            # del inner_decoder_p_vec
            # del inner_aligner_params
            # del inner_decoder_params
            # copy inner_grads to main network
            for i in range(self.num_particles):
                for p_tar, p_src in zip(model_encoder_params[i],
                                        BayesModelAgnosticMetaLearning.vector_to_list_params(inner_grads[i],
                                                                                             model_encoder_params[i])):
                    p_tar.grad.data.add_(p_src) # todo: divide by num_of_sample if inner is in ba
            # copy aligner grads to the main network
            for p_tar, p_src in zip(model_aligner_params,
                            BayesModelAgnosticMetaLearning.vector_to_list_params(alinger_grads_vec, model_aligner_params)):
                p_tar.grad.data.add_(p_src)
            # copy decoder grads to the main network
            for p_tar, p_src in zip(model_decoder_params,
                            BayesModelAgnosticMetaLearning.vector_to_list_params(decoder_grads_vec, model_decoder_params)):
                p_tar.grad.data.add_(p_src)
            # trying to free gpu memory 
            # del inner_grads
            # del alinger_grads_vec
            # del decoder_grads_vec
            # not sure it would help
            # del kernel_matrix
            # del grad_kernel
            # del distance_nll
            # del inner_grads
            # torch.cuda.empty_cache()
            
        logger.info(f"Inner loss: {sum(inner_loss)/self.num_particles}")
        # accumulate to compute mean over particles
        loss_over_pars = []
        bert_grad_outer = None
        aligner_grad_outer = None 
        decoder_grad_outer = None
        for i in range(self.num_particles):
            mean_outer_loss = torch.Tensor([0.0]).to(self.device)
            for outer_batch in outer_batches:
                enc_outer_list = [enc_input for enc_input, dec_output in outer_batch]
                column_pointer_maps = [
                {i: [i] for i in range(len(desc["columns"]))} for desc in enc_outer_list
                ]
                table_pointer_maps = [
                    {i: [i] for i in range(len(desc["tables"]))} for desc in enc_outer_list
                ]
                plm_output = model.bert_model(enc_outer_list)
                enc_states = []
                for idx, (enc_input, plm_out) in enumerate(zip(enc_outer_list, plm_output)):
                    relation = model.schema_linking(enc_input)
                    (
                        q_enc_new_item,
                        c_enc_new_item,
                        t_enc_new_item,
                    ) = inner_encoders[i](enc_input, 
                                                        plm_out,
                                                        relation)
                    # attention memory 
                    memory = []
                    include_in_memory = inner_encoders[0].include_in_memory
                    if "question" in include_in_memory:
                        memory.append(q_enc_new_item)
                    if "column" in include_in_memory:
                        memory.append(c_enc_new_item)
                    if "table" in include_in_memory:
                        memory.append(t_enc_new_item)
                    memory = torch.cat(memory, dim=1)
                    # alignment matrix
                    align_mat_item = inner_aligner(
                        enc_input, q_enc_new_item, c_enc_new_item, t_enc_new_item, relation
                    )
                    enc_states.append(
                        SpiderEncoderState(
                            state=None,
                            words_for_copying=enc_input["question_for_copying"],
                            tokenizer=inner_encoders[0].tokenizer,
                            memory=memory,
                            question_memory=q_enc_new_item,
                            schema_memory=torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                            pointer_memories={
                                "column": c_enc_new_item,
                                "table": t_enc_new_item,
                            },
                            pointer_maps={
                                "column": column_pointer_maps[idx],
                                "table": table_pointer_maps[idx],
                            },
                            m2c_align_mat=align_mat_item[0],
                            m2t_align_mat=align_mat_item[1],
                        )
                    )
                losses = []
                for enc_state, (enc_input, dec_output) in zip(enc_states, outer_batch):
                    ret_dic = inner_decoder(dec_output, enc_state)
                    losses.append(ret_dic["loss"])
                loss = torch.mean(torch.stack(losses, dim=0), dim=0)
                mean_outer_loss += loss
            mean_outer_loss.div_(len(outer_batches)*self.num_particles)
            # compute gradients of outer loss
            outer_grads = autograd.grad(mean_outer_loss, 
                                        model_bert_params
                                        + inner_encoder_params[i] 
                                        + inner_aligner_params 
                                        + inner_decoder_params,
                                        allow_unused=True)
            
            for idx, (p_tar, p_src) in enumerate(zip(model_encoder_params[i],
                                    outer_grads[bert_model_len:bert_model_len
                                                +particle_len])):
                if p_src is not None:
                    p_tar.grad.data.add_(p_src)
                else:
                    p_tar.grad.data.add_(torch.zeros_like(model_encoder_params[i][idx]))
            if bert_grad_outer is None:
                bert_grad_outer = outer_grads[:bert_model_len]
                aligner_grad_outer = outer_grads[bert_model_len
                                        +particle_len:bert_model_len
                                        +particle_len
                                        +aligner_len]
                decoder_grad_outer = outer_grads[bert_model_len
                                            +particle_len
                                            +aligner_len:]
            else: 
                bert_grad_outer += outer_grads[:bert_model_len]
                aligner_grad_outer += outer_grads[bert_model_len
                                        +particle_len:bert_model_len
                                        +particle_len
                                        +aligner_len]
                decoder_grad_outer += outer_grads[bert_model_len
                                            +particle_len
                                            +aligner_len:]
            loss_over_pars.append(mean_outer_loss.item())
        # copy inner_grads to main network
        for idx, (p_tar, p_src) in enumerate(zip(model_bert_params,
                                bert_grad_outer)):
            if p_src is not None:
                p_tar.grad.data.add_(p_src)
            else:
                p_tar.grad.data.add_(torch.zeros_like(model_bert_params[idx]))
            
        # copy aligner grads to the main network
        for idx, (p_tar, p_src) in enumerate(zip(model_aligner_params,
                                aligner_grad_outer)):
            if p_src is not None:
                p_tar.grad.data.add_(p_src)
            else:
                p_tar.grad.data.add_(torch.zeros_like(model_aligner_params[idx]))
        # copy decoder grads to the main network
        for idx, (p_tar, p_src) in enumerate(zip(model_decoder_params,
                                decoder_grad_outer)):
            if p_src is not None:
                p_tar.grad.data.add_(p_src)
            else:
                p_tar.grad.data.add_(torch.zeros_like(model_decoder_params[idx]))
        logger.info(f"Outer loss: {sum(loss_over_pars)}")
            # Compute loss on udpated inner model
            # mean_outer_loss = torch.Tensor([0.0]).to(self.device)
            # for outer_batch in outer_batches:
            #     outer_ret_dict = inner_model(outer_batch)
            #     mean_outer_loss += outer_ret_dict["loss"]
            # mean_outer_loss.div_(len(outer_batches))
            # logger.info(f"Outer loss: {mean_outer_loss.item()}")
        
        
        
        # for p, g_o in zip(model.parameters(), grad_outer):
        #         if g_o is not None:
        #             p.grad.data.add_(g_o.data)
                    
        # del grad_outer
        final_loss = (sum(inner_loss) + sum(loss_over_pars))/self.num_particles
        ret_dic["loss"] = final_loss
        # del inner_encoders
        del inner_aligner
        # del inner_decoder
        import gc
        gc.collect()
        # torch.cuda.empty_cache()
        
        return ret_dic
    
    @staticmethod
    def get_kernel(params: torch.Tensor, num_of_particles):
        """
        Compute the RBF kernel for the input
        
        Args:
            params: a tensor of shape (N, M)
        
        Returns: kernel_matrix = tensor of shape (N, N)
        """
        pairwise_d_matrix = BayesModelAgnosticMetaLearning.get_pairwise_distance_matrix(x=params)

        median_dist = torch.quantile(input=pairwise_d_matrix, q=0.5)  # tf.reduce_mean(euclidean_dists) ** 2
        h = median_dist / np.log(num_of_particles)

        kernel_matrix = torch.exp(-pairwise_d_matrix / h)
        kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
        grad_kernel = -torch.matmul(kernel_matrix, params)
        grad_kernel += params * kernel_sum
        grad_kernel /= h

        return kernel_matrix, grad_kernel, h
    
    @staticmethod
    def get_kernel_wSGLD_B(params: torch.Tensor, num_of_particles):
        """
        Compute the RBF kernel and repulsive term for wSGLD 
        
        Args:
            params: a tensor of shape (N, M)
        
        Returns: 
            - kernel_matrix = tensor of shape (N, N)
            - repulsive term 
        """
        pairwise_d_matrix = BayesModelAgnosticMetaLearning.get_pairwise_distance_matrix(x=params)
        median_dist = torch.quantile(input=pairwise_d_matrix, q=0.5)
        h = median_dist / np.log(num_of_particles)
        kernel_matrix = torch.exp(-pairwise_d_matrix / h)
        kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
        # compute repulsive term of w_SGLD_B 
        # invert of kernel_sum Nx1 
        invert_kernel_sum = kernel_sum.pow_(-1) 
        grad_kernel = params*(torch.matmul(kernel_matrix, invert_kernel_sum) +
                                torch.sum(kernel_matrix*invert_kernel_sum, dim=1, keepdim=True))
        grad_kernel += -(torch.matmul(kernel_matrix*torch.transpose(invert_kernel_sum,0,1), params) +
                         torch.matmul(kernel_matrix, params)*invert_kernel_sum)
        grad_kernel /= h
        
        return grad_kernel, h
    
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
    
    @staticmethod
    def vector_to_list_params(vector, other_params):
    
        params = []
        
        # pointer for each layer params
        pointer = 0 
        
        for param in other_params:
            # total number of params each layer
            num_params = int(np.prod(param.shape))
            
            params.append(vector[pointer:pointer+num_params].view(param.shape))
            
            pointer += num_params

        return params
    
BMAML = BayesModelAgnosticMetaLearning

