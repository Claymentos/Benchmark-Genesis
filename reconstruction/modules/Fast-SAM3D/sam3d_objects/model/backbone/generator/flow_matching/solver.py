# Copyright (c) Meta Platforms, Inc. and affiliates.
import optree
import torch
from functools import partial
import time
from sam3d_objects.data.utils import tree_tensor_map
import numpy as np
import torch
import torch.nn.functional as F


def linear_approximation_step(x_t, dt, velocity):
    # x_tp1 = x_t + velocity * dt
    x_tp1 = tree_tensor_map(lambda x, v: x + v * dt, x_t, velocity)
    return x_tp1


def gradient(output, x, create_graph: bool = False):
    tensors, pyspec = optree.tree_flatten(
        x, is_leaf=lambda x: isinstance(x, torch.Tensor)
    )
    grad_outputs = [torch.ones_like(output).detach() for _ in tensors]
    grads = torch.autograd.grad(
        output,
        tensors,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
    )
    return optree.tree_unflatten(pyspec, grads)


def simple_dynamics(x, t):
    return -x + torch.sin(t)


class ODESolver:
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        raise NotImplementedError

    def solve_iter(self, dynamics_fn, x_init, times, *args, **kwargs):
        x_t = x_init
        for t0, t1 in zip(times[:-1], times[1:]):
            dt = t1 - t0
            x_t, v = self.step(dynamics_fn, x_t, t0, dt, *args, **kwargs)
            yield x_t, t0 ,v

    def solve(self, dynamics_fn, x_init, times, *args, **kwargs):
        for x_t, _, _, in self.solve_iter(dynamics_fn, x_init, times, *args, **kwargs):
            pass
        return x_t
    

# https://en.wikipedia.org/wiki/Euler_method
class Euler(ODESolver):
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        # 速度
        velocity = dynamics_fn(x_t, t, *args, **kwargs)
        x_tp1 = linear_approximation_step(x_t, dt, velocity)
        return x_tp1,velocity


# https://arxiv.org/abs/2505.05470
class SDE(ODESolver):
    def __init__(self, **kwargs):
        super().__init__()
        self.sde_strength = kwargs.get("sde_strength", 0.1)

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        velocity = dynamics_fn(x_t, t, *args, **kwargs)
        sigma = 1 - t
        var_t = sigma / (1 - torch.tensor(sigma).clamp(min=dt))
        std_dev_t = (
            torch.sqrt(variance) * self.sde_strength
        )  # self.sde_strength = alpha

        def compute_mean(x, v):
            drift_term = x * (std_dev_t**2 / (2 * sigma) * dt)
            velocity_term = v * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            return x + drift_term + velocity_term

        prev_sample_mean = tree_tensor_map(compute_mean, x_t, velocity)

        # Generate noise and compute final sample using tree_tensor_map
        def add_noise(mean_val):
            variance_noise = torch.randn_like(mean_val)
            return mean_val + std_dev_t * torch.sqrt(torch.tensor(dt)) * variance_noise

        prev_sample = tree_tensor_map(add_noise, prev_sample_mean)

        return prev_sample


# https://en.wikipedia.org/wiki/Midpoint_method
class Midpoint(ODESolver):
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        half_dt = 0.5 * dt

        x_mid = Euler.step(self, dynamics_fn, x_t, t, half_dt, *args, **kwargs)

        velocity_mid = dynamics_fn(x_mid, t + half_dt, *args, **kwargs)
        x_tp1 = linear_approximation_step(x_t, dt, velocity_mid)
        return x_tp1,velocity_mid


# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
class RungeKutta4(ODESolver):

    def k1(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        return dynamics_fn(x_t, t, *args, **kwargs)

    def k2(self, dynamics_fn, x_t, t, dt, k1, *args, **kwargs):
        x_k1 = linear_approximation_step(x_t, dt * 0.5, k1)
        return dynamics_fn(x_k1, t + dt * 0.5, *args, **kwargs)

    def k3(self, dynamics_fn, x_t, t, dt, k2, *args, **kwargs):
        x_k2 = linear_approximation_step(x_t, dt * 0.5, k2)
        return dynamics_fn(x_k2, t + dt * 0.5, *args, **kwargs)

    def k4(self, dynamics_fn, x_t, t, dt, k3, *args, **kwargs):
        x_k3 = linear_approximation_step(x_t, dt, k3)
        return dynamics_fn(x_k3, t + dt, *args, **kwargs)

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        k1 = self.k1(dynamics_fn, x_t, t, dt, *args, **kwargs)
        k2 = self.k2(dynamics_fn, x_t, t, dt, k1, *args, **kwargs)
        k3 = self.k3(dynamics_fn, x_t, t, dt, k2, *args, **kwargs)
        k4 = self.k4(dynamics_fn, x_t, t, dt, k3, *args, **kwargs)

        def compute_velocity(k1, k2, k3, k4):
            return (k1 + 2 * k2 + 2 * k3 + k4) / 6

        velocity_k = tree_tensor_map(compute_velocity, k1, k2, k3, k4)
        x_tp1 = linear_approximation_step(x_t, dt, velocity_k)
        return x_tp1,velocity_k


from cache_utils_slat_end import cal_type,cache_init
class Euler_end_slat(ODESolver):
    def __init__(self, thresh=0.0, dir_weight=0.5, ret_steps=1, full_steps=25,carving_ratio = 0.0):
       
        super().__init__()
        self.thresh = thresh 
        self.dir_weight =  dir_weight
        self.ret_steps = ret_steps
        self.full_steps = full_steps
        self.carving_ratio = carving_ratio
       
        self.cache_dic = None
        self.current = None

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
     
        should_calc = True
        should_calc = cal_type(self.cache_dic, self.current ,x_t)
        if should_calc:
            if self.current['is_f3c_active'] and self.current['use_f3c']:
                coords_scores = self.stability_tracker.coords_scores
                self.current['cached_indices'],  self.current['fast_update_indices'] = self.stability_tracker.update_and_select_combined(self.cache_dic['cache']['prev_v'], self.current['num_to_skip'],t=0, coords_scores = coords_scores,spatial_weight=0.3)

                step_args_list= list(args)
                x_input = x_t[:, self.current['fast_update_indices'], :] if self.current['is_f3c_active'] else x_t

                if len(step_args_list) > 1:
                    full_coords = self.full_coords_backup
                    idx_np = self.current['fast_update_indices'].detach().cpu().numpy()
                    cropped_coords = full_coords[idx_np].astype(np.int32)
                    step_args_list[1] = cropped_coords
                            
                step_args = tuple(step_args_list)
                velocity = dynamics_fn(x_input, t, *step_args, **kwargs)


            else:

                velocity = dynamics_fn(x_t, t, *args, **kwargs)
            
            if self.current['is_f3c_active'] and self.current['use_f3c']:
                final_v_tokens = self.cache_dic['cache']['prev_v'].clone() 
                final_v_tokens[:, self.current['fast_update_indices'], :] = velocity.to(final_v_tokens.dtype)
                velocity = final_v_tokens
           
            
            prev_x = self.cache_dic['cache']['prev_x']
            prev_prev_x = self.cache_dic['cache']['prev_prev_x']
            prev_v = self.cache_dic['cache']['prev_v']
            k = self.cache_dic['cache']['k']

            if prev_x is not None and prev_prev_x is not None:
                output_change = (velocity - prev_v).abs().mean()
                prev_input_change = (prev_x - prev_prev_x).abs().mean() + 1e-8
                current_k = output_change / prev_input_change
                
                if k is None:
                    self.cache_dic['cache']['k'] = current_k
                else:
                    self.cache_dic['cache']['k'] = 0.7 * k + 0.3 * current_k 

     
            if prev_x is not None:
                self.cache_dic['cache']['prev_prev_x'] = prev_x
            self.cache_dic['cache']['prev_x'] = x_t.detach().clone()
            self.cache_dic['cache']['prev_v'] = velocity.detach().clone()
            self.cache_dic['cache']['easy'] = velocity - x_t
        else:

            velocity = x_t + self.cache_dic['cache']['easy']
    
            self.cache_dic['cache']['prev_x'] = x_t.detach().clone()
            self.cache_dic['cache']['prev_v'] = velocity.detach().clone()

        x_tp1 = linear_approximation_step(x_t, dt, velocity)
        self.current['step'] += 1
        
        return x_tp1, velocity

    def solve_iter(self, dynamics_fn, x_init, times, LEADER, stability_tracker, *args, **kwargs):

        self.cache_dic ,self.current = cache_init(self.full_steps)
        self.cache_dic['thresh'] = self.thresh 
        self.cache_dic['dir_weight'] =  self.dir_weight
        self.cache_dic['first_enhance'] = self.ret_steps

        self.LEADER = LEADER    
        self.stability_tracker = stability_tracker

        current_args_list = list(args) 
        if len(current_args_list) > 1:
            self.full_coords_backup = current_args_list[1] 
        
        B, N, C = x_init.shape
        LEADER.total_tokens = N
        LEADER.schedule_is_set = True
        self.last_coords =  current_args_list[1]

        x_t = x_init
        for t0, t1 in zip(times[:-1], times[1:]):
            
            cache = self.cache_dic['cache']
            self.current['is_f3c_active'] = False  
            current_step = LEADER.current_step

            if self.current['use_f3c'] and  cache['prev_v']is not None and current_step >= LEADER.full_sampling_steps:
                self.current['num_to_skip'] = int(self.carving_ratio * N)
                if self.current['num_to_skip'] > 0 and self.current['num_to_skip'] < N:
                    self.current['is_f3c_active'] = True

            dt = t1 - t0
            x_t, v = self.step(dynamics_fn, x_t, t0, dt, *args, **kwargs)

            LEADER.increase_step()
            yield x_t, t0, v
            
