import torch
import torch.nn.functional as F

def cal_type(cache_dic, current, input):

    cache = cache_dic['cache'] 
    x_t = input
    is_first_steps = (current['step'] < cache_dic['first_enhance'])
    should_calc = True

    if is_first_steps:
        should_calc = True
    else:
        if cache['prev_x'] is not None and cache['prev_v'] is not None and cache['prev_prev_x'] is not None:
        
            delta_x = x_t - cache['prev_x']
            input_change_mag = delta_x.abs().mean()
            
    
            prev_delta_x = cache['prev_x'] - cache['prev_prev_x']
            cos_sim = F.cosine_similarity(delta_x.reshape(1, -1), 
                                            prev_delta_x.reshape(1, -1), 
                                            dim=1)
            direction_error = (1.0 - cos_sim).item() 

            if cache['k'] is not None:
                output_norm = cache['prev_v'].abs().mean() + 1e-6
                mag_error = cache['k'] * (input_change_mag / output_norm)
                # current_step_error = mag_error + cache['dir_weight'] * direction_error
                current_step_error = mag_error
                cache['error'] += current_step_error
                
                if cache['error'] < cache_dic['thresh']:
                    should_calc = False
                    
                else:
                    should_calc = True
    
            else:
                should_calc = True
                
        else:
            should_calc = True

    if should_calc:
        cache['error'] = 0 
        current['type']  = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])

    else:
        current['type'] = 'Taylor'
        cache_dic['cache_counter'] += 1

    return should_calc
        



