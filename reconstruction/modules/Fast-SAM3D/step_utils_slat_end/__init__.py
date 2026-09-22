from typing import Dict 
import torch
import math


def derivative_approximation(cache_dic: Dict, current: Dict, feature):
    """
    Compute derivative approximation.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature 

    if current['activated_steps'][-1] == 0:
        cache_dic['cache']['feature'] = updated_taylor_factors
        return 

    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    for i in range(cache_dic['max_order']):
        if (cache_dic['cache']['feature'] is not None and cache_dic['cache']['feature'].get(i, None) is not None):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache']['feature'][i]) / difference_distance
        else:
            break

    cache_dic['cache']['feature'] = updated_taylor_factors



def step_formula(cache_dic: Dict, current: Dict): 
    x = current['step'] - current['activated_steps'][-1]
    cache = cache_dic['cache']['feature']
    output = 0

    for i in range(len(cache)):
        term =  (1 / math.factorial(i)) * cache[i] * (x ** i)
        output += term

    return output




def step_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache']['feature'] = {}
