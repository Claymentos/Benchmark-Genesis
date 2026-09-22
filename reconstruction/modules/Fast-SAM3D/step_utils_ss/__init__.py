
import torch
import math
from typing import Dict, Union, List


def derivative_approximation(cache_dic: Dict, current: Dict, feature: Union[Dict[str, torch.Tensor], torch.Tensor]):
    if len(current['activated_steps']) < 2:
        difference_distance = 1.0 
    else:
        difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    if isinstance(feature, torch.Tensor):
        feature = {'default': feature}

    prev_module_cache = cache_dic['cache'][-1][current['layer']][current['module']]
    updated_module_factors = {} 

    for key, tensor_val in feature.items():
        current_key_factors = {}
        current_key_factors[0] = tensor_val 

        prev_key_cache = prev_module_cache.get(key, None) if prev_module_cache else None
        
        if current['step'] > 0 and prev_key_cache is None:
            # print(f"⚠️ Step {current['step']} (Module: {current['module']}) | Key '{key}' Loss cache")
            pass


        for i in range(cache_dic['max_order']):
            has_prev = (prev_key_cache is not None) and (i in prev_key_cache)
            is_within = current['step'] < (current['num_steps'] - cache_dic['first_enhance'] + 1)
            if has_prev and is_within:
                prev_val = prev_key_cache[i]
                current_val = current_key_factors[i] 
                current_key_factors[i + 1] = (current_val - prev_val) / difference_distance
            else:
                # print(f"🛑 Stop: Step={current['step']}, Order={i+1}, Reason: PrevExist={has_prev},is_within{is_within}")
                break
        
        updated_module_factors[key] = current_key_factors


    if current['layer'] not in cache_dic['cache'][-1]:
        cache_dic['cache'][-1][current['layer']] = {}
        
    cache_dic['cache'][-1][current['layer']][current['module']] = updated_module_factors

    # print(f"💾 Step {current['step']} 缓存写入完成。Keys: {list(updated_module_factors.keys())}")


def step_formula(cache_dic: Dict, current: Dict, prev_v :Dict, beta = 0.5) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: 
    x = current['step'] - current['activated_steps'][-1]
    module_cache = cache_dic['cache'][-1][current['layer']][current['module']]

    def compute_single_expansion(factors_dict, x_dist):
        result = 0
        for i in range(len(factors_dict)):
    
            term = (1 / math.factorial(i)) * factors_dict[i] * (x_dist ** i)
            result += term
        return result
    
    def compute_single_expansion_ema(factors_dict, x_dist, prev_value, beta = 0.5):
        result = 0
        for i in range(len(factors_dict)):
            term = (1 / math.factorial(i)) * factors_dict[i] * (x_dist ** i)
            result += term

        result = beta * prev_value + (1.0 - beta) * result
        return result

    def get_raw(factors_dict, x_dist):
        result = 0
        result = factors_dict[0]* (x_dist ** 0)
        return result

    first_val = next(iter(module_cache.values()))

    if isinstance(first_val, dict):
        output_dict = {}
        for key, factors in module_cache.items():

            full_keys = ['shape']
            ema_keys = ['6drotation_normalized','scale','translation','translation_scale'] 

            if key in full_keys:
                output_dict[key] = compute_single_expansion(factors, x)
            if key in ema_keys:
                prev_value = prev_v[key]
                output_dict[key] = compute_single_expansion_ema(factors, x, prev_value, beta)


            output_dict[key] = compute_single_expansion(factors, x)
         
        return output_dict
    else:
   
        return compute_single_expansion(module_cache, x)


def step_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and expand storage for different-order derivatives.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    if current['step'] == (current['num_steps'] - 1):
        pass
