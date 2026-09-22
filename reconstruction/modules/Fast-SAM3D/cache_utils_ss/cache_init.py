
def cache_init(num_steps , cache_interval = 3 , max_order = 1 , first_enhance = 2 , end_enhance = 24):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache[-1]={}

    cache_dic['cache_counter'] = 0

    cache[-1]['final'] = {}
    cache[-1]['final']['final'] = {}
    cache[-1]['final']['final']['final'] = {}

    cache_dic['cache'] = cache
    
    cache_dic['cache_interval'] = cache_interval
    cache_dic['max_order'] = max_order
    cache_dic['first_enhance'] = first_enhance
    cache_dic['end_enhance'] = end_enhance

    cache_dic['taylor_cache'] = True
    
    current = {}
    current['activated_steps'] = []

    current['step'] = 0
    current['num_steps'] = num_steps

    return cache_dic, current
