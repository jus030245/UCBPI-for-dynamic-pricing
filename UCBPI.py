# In[1]:


# import modules 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

def type5(size):
    output = []
    for i in range(size):
        seed = np.random.rand()
        if seed < 0.3:
            output.append(0.2)
        else:
            output.append(0.9)
    return np.array(output)

class ucb_bandit:
    '''
    Upper Confidence Bound Bandit
    
    Inputs 
    ============================================
    k: number of arms (int)
    c: constant for exploration parameter (hyperparameter)
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''
    def __init__(self, prices, c, iters, seg_num, seg_size):
        # list of interested prices
        self.prices = prices
        # Number of arms
        self.k = len(self.prices)
        # Exploration parameter
        self.c = c
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(self.k)
        # Total mean profit
        self.mean_profit = 0
        self.profit = np.zeros(iters)
        # Mean reward for each arm
        self.k_profit = np.zeros(self.k)
        # The amount of known segments
        self.seg_num = seg_num
        # The known size of each segments
        self.seg_size = seg_size
        
        # Demand learning
        self.nu_s_t = np.repeat((min(prices)+max(prices))/2, seg_num)
        self.delta = np.repeat((min(prices)+max(prices))/2, seg_num)
        self.p_min = np.repeat(min(prices), seg_num) 
        self.p_max = np.repeat(max(prices), seg_num)
        self.bias = 0
        self.delta_hat = max(self.delta)

            
    def valuation_nu_generater(self,between_seg_dis='type1'):
        #simulate the true mu of each segment from certain distribution
        if between_seg_dis == 'type1':
            #beta(2,9)
            self.nu_true = np.random.beta(2,9,self.seg_num)
        elif between_seg_dis == 'type2':
            #beta(2,2)
            self.nu_true = np.random.beta(2,2,self.seg_num)
        elif between_seg_dis == 'type3':   
            #beta(9,2)
            self.nu_true = np.random.beta(9,2,self.seg_num)
        elif between_seg_dis == 'type4':   
            #bimodal continuous beta(0.2,0.3)
            self.nu_true = np.random.beta(0.2,0.3,self.seg_num)
        elif between_seg_dis == 'type5':
            #discontinuous finite mixture model with 70% 0.2 and 30% 0.9
            self.nu_true = type5(self.seg_num)
    
    def customer_simulation(self,customer=10,delta_true='default'):
        #assume that we know that the coming customer are from which segments
        #only change price after 10 customer
        self.customer = customer
        if delta_true == 'default':
            self.delta_true = (max(self.prices)-min(self.prices))/self.k*0.2
        else:
            self.delta_true = delta_true
        sampled_segments = np.random.choice(range(self.seg_num),customer)
        with_in_variation = np.random.uniform(-self.delta_true,self.delta_true,customer)
        v_s_i = self.nu_true[sampled_segments] + with_in_variation
        return sampled_segments, v_s_i
    
    def pull(self, sampled_segments, v_s_i, PI=True, original=False):
        # Select action according to UCB Criteria
        # The main strategy
        # argmax opt for the first one for duplicated values
        # UCB original or amended
        if original == True:
            criteria = (self.k_profit + np.sqrt( self.c * (np.log(self.n)) / self.k_n))
        else:
            criteria = (self.k_profit + self.prices * np.sqrt( self.c * (np.log(self.n)) / self.k_n))
        # Partial Intification
        if PI == True:
            LB = []
            UB = []
            for price in self.prices:
                LB_k = price * (self.seg_size*(self.p_min > price)).sum()
                UB_k = price * (self.seg_size*(self.p_max > price)).sum()
                LB.append(LB_k)
                UB.append(UB_k)
            criteria[np.array(UB) <= max(LB)] = 0
        else:
            pass
        
        a = np.argmax(criteria)
        chosen_price = prices[a]
        profit = chosen_price * (v_s_i > prices[a]).sum()
        
        # Update price bounds for segments
        self.p_max[sampled_segments[v_s_i > chosen_price]] = np.minimum(self.p_max[sampled_segments[v_s_i > chosen_price]],
                                                      v_s_i[v_s_i > chosen_price])
        self.p_min[sampled_segments[v_s_i < chosen_price]] = np.maximum(self.p_min[sampled_segments[v_s_i < chosen_price]],
                                                      v_s_i[v_s_i < chosen_price])
        # Update nu_s_t
        self.nu_s_t = (self.p_min+self.p_max)/2
        # Update delta
        self.delta = (self.p_max-self.p_min)/2
        # Updata bias
        self.bias = 0
        # Update delta_hat
        self.delta_hat = max(self.delta) + self.bias
        
        # Update total
        self.mean_profit = self.mean_profit + (profit - self.mean_profit) / self.n
        
        # Update results for a_k
        self.k_profit[a] = self.k_profit[a] + (profit - self.k_profit[a]) / self.k_n[a]
        
        # Update counts (adjust the order of counts update)
        self.n += 1
        self.k_n[a] += 1
        
    def run(self,customer=10,PI=True,original=False):
        for i in range(self.iters):
            sampled_segments, v_s_i = self.customer_simulation(customer=customer)
            self.pull(sampled_segments=sampled_segments, v_s_i=v_s_i,PI=PI,original=original)
            self.profit[i] = self.mean_profit
            
    def reset(self, mu=None):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_profit = 0
        self.profit = np.zeros(self.iters)
        self.k_profit = np.zeros(self.k)
        self.nu_s_t = np.repeat((min(prices)+max(prices))/2, seg_num)
        self.delta = np.repeat((min(prices)+max(prices))/2, seg_num)
        self.p_min = np.repeat(min(prices), seg_num) 
        self.p_max = np.repeat(max(prices), seg_num)
        self.bias = 0
        self.delta_hat = max(self.delta)
        
    def regret(self):
        demand = []
        for price in prices:
            demand.append((self.seg_size * (price < self.nu_true)).sum())
        expected_profit = max(self.prices * np.array(demand)) * self.customer
        regret = 1 - (self.profit[-1] / expected_profit)
        return regret
