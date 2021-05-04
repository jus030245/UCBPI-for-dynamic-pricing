###DEMO###
import numpy as np
import pandas
import matplotlib.pyplot as plt
import UCBPI

#experiments set up
prices = np.linspace(0,1,101)[1:]
seg_num = 100000
segment_size = np.repeat(1/seg_num,seg_num)
iters = 100
c = 2


# In[3]:


#create the MAB environment
ucb = ucb_bandit(prices, c, iters, seg_num,segment_size)


# In[4]:


#choose the distribution of mean of the segment valuation 
#can be choosen from type1 to type5
#type four is a bimodal distribution
ucb.valuation_nu_generater('type4')
plt.hist(ucb.nu_true)


# In[5]:


# #start pricing and learning
# episodes = 5
# ucb_profit_concat = np.zeros(iters)
# import time
# start = time.time()
# for i in range(episodes):
#     ucb.reset()
#     ucb.run()
#     ucb_profit_concat = ucb_profit_concat + (ucb.profit - ucb_profit_concat) / (i + 1)
    
# end = time.time()
# print('Time Spent:',end-start)
# the average profit mean over iterations
# plt.figure(figsize=(24,16))
# plt.plot(ucb_profit_concat)


# In[6]:


import time
start = time.time()
ucb.run()
end = time.time()
print('Time Spent:',end-start)


# In[7]:


plt.figure(figsize=(24,16))
plt.bar(prices,ucb.k_n,width=0.01,label='UCB')


# In[8]:


###Parameters###
#The number of times the kth price was chosen
print('The number of times the kth price was chosen:',ucb.k_n)
#The mean profit over iterations
print('The mean profit over iterations:',ucb.profit)
#The percerived profit for each price at the end
print('Perceived profit for each price:',ucb.k_profit)

#The valuation midpoint drawn from typed distribution
print('True Between Segments means:',ucb.nu_true)
#The observed midpoint
print('Observed Between Segments means:',ucb.nu_s_t)
#Inserted delta
print('True Delta:',ucb.delta_true)
#Observed delta
print('True Delta:',ucb.delta_hat)


# In[9]:


#calculating loss
print(ucb.regret())


# In[10]:


###Comparison###
#UCB vs UCB-amended vs UCB-PI
#setting up same environment
prices = np.linspace(0,1,101)[1:]
seg_num = 1000
segment_size = np.repeat(1/seg_num,seg_num)
iteration = 100000
c = 2
UCB = ucb_bandit(prices, c, iteration, seg_num,segment_size)
UCB_amended = ucb_bandit(prices, c, iteration, seg_num,segment_size)
UCB_PI = ucb_bandit(prices, c, iteration, seg_num,segment_size)
UCB.valuation_nu_generater('type4')
UCB_amended.nu_true = UCB.nu_true
UCB_PI.nu_true = UCB.nu_true


# In[11]:


#different strategies
UCB.run(PI=False,original=True)
UCB_amended.run(PI=False)
UCB_PI.run()


# In[12]:


print(UCB.regret())
print(UCB_amended.regret())
print(UCB_PI.regret())


# In[13]:


plt.figure(figsize=(12,8))
plt.plot(UCB.profit, label='UCB')
plt.plot(UCB_amended.profit, label='UCB_amenden')
plt.plot(UCB_PI.profit, label='UCB_PI')
plt.legend(bbox_to_anchor=(1.3, 0.5))
plt.xlabel("Iterations")
plt.ylabel("Average Profit")
plt.title("Average rewards for UCB vs UCB_amended vs UCB_PI")
plt.show()

