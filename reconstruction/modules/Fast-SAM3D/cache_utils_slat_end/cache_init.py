
def cache_init(num_steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}

    cache['k'] = None      
    cache['prev_x'] = None 
    cache['prev_v'] = None 
    cache['prev_prev_x'] = None 
    cache['error'] = None
    cache['feature'] = None 
    cache['easy'] = None 

    cache_dic['thresh'] = 1.0       
    cache_dic['dir_weight'] =  0.5  
    cache_dic['cache_counter'] = 0 
    cache_dic['cache'] = cache     
    cache_dic['taylor_cache'] = True 

    mode = 'fast' #['fast', 'mid', 'detailed']

    if mode == 'fast': 
        cache_dic['max_order'] = 2      
        cache_dic['first_enhance'] = 1  

    elif mode == 'mid':
        cache_dic['max_order'] = 2
        cache_dic['first_enhance'] = 3
        
    elif mode == 'detailed':
        cache_dic['max_order'] = 2
        cache_dic['first_enhance'] = 3
    
    cache_dic['taylor_cache'] = True
    
    current = {}
    current['type'] = None
    current['activated_steps'] = [] 
    current['step'] = 0
    current['num_steps'] = num_steps
  
    current['use_f3c'] = True
    current['is_f3c_active'] = False
    current['num_to_skip'] = 0
    current['cache_indices'] = None
    current['fast_update_indices'] = None



    return cache_dic, current
