# -*- coding: utf-8 -*-
"""
Created on Fri May 20 18:03:22 2016

@author: Shashank
"""
import pandas as pd
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import heapq



##################### constants
num_movies = 1682
num_features = 19
num_users = 943


################################### functions ###########################################################

####### function to compute the posterior probabilities of an item  given its parent features
def item_prob_func(item):
   par_feat = np.nonzero(movie_feat_mat.ix[item,:])[0]
   prob = 0
   
   M = 0
   for feat in par_feat:
     nk = len(np.nonzero(movie_feat_mat.ix[:,feat])[0])
     M = M + math.log((num_movies/nk) + 1)
     
   for feat in par_feat:
       nk = len(np.nonzero(movie_feat_mat.ix[:,feat])[0])
       for state in range(0,2):
           if(state == 0):
              wt =  0
           else:
              wt = math.log((num_movies/nk) + 1) * (1/M)
             
           prob = prob + wt * bn.node[feat_nodes[feat]]['prob'][state] 

   bn.node[item_nodes[item]]['prob'][1]  = prob
   bn.node[item_nodes[item]]['prob'][0]  = 1 - prob
    


####### function to compute the posterior probabilities of a user-state combination given its parent items #######
def user_prob_func(user,p):
   par = np.nonzero(user_rat_mat.ix[user,:])[0]
   prob = 0
     
   for itm in par:
       rat = user_rat_mat.ix[user,itm]
       for state in range(0,2):
           if( (state == 1) and (p == rat) ):
              wt =  1 / len(par)
           elif((state == 1) and (p != rat)):
              wt = 0
           elif((state == 0) and (p == 0)):
              wt = 1 / len(par)
           elif((state == 0) and (p != 0)):
              wt = 0
              
           prob = prob + wt * bn.node[item_nodes[itm]]['prob'][state] 

   bn.node[user_nodes[user]]['prob'][p]  = prob
  


####### function to compute the posterior probabilities of a a_cb-state combination given its parent items #######
def acb_prob_func(p):
   par = item_rat_act_user 
   prob = 0
     
   for itm in par:
       rat = user_rat_mat.ix[active_user,itm]
       for state in range(0,2):
           if( (state == 1) and (p == rat) ):
              wt =  1 / len(par)
           elif((state == 1) and (p != rat)):
              wt = 0
           elif((state == 0) and (p == 0)):
              wt = 1 / len(par)
           elif((state == 0) and (p != 0)):
              wt = 0
              
           prob = prob + wt * bn.node[item_nodes[itm]]['prob'][state] 

   bn.node["a_cb"]['prob'][p]  = prob


####### function to compute the posterior probabilities of a a_cf-state combination given its parent users #######
def cf_prob_func(p):
   par =  users_sim_act_user_nhm
   prob = 0
     
   for usr in par:
       rel_sim = sim_nhm[active_user][usr]/tot_sim
       rat_usr = np.nonzero(user_rat_mat.ix[usr,:])[0]
       int_set = np.intersect1d(rat_usr,item_rat_act_user)
             
       for state in range(0,6):
           N_both = list()       
           for itm in int_set:
                if( (user_rat_mat.ix[active_user,itm] == p) and (user_rat_mat.ix[usr,itm] == state) ):          
                       N_both.append(itm)
           N_num = len(N_both) 

           N_b = list()       
           for itm in int_set:
                if((user_rat_mat.ix[usr,itm] == state) ):          
                       N_b.append(itm)
           N_den = len(N_b) 
           
           Pr = (N_num + (1/5))/(N_den + 1)
           
           if( (state != 0) and (p != 0) ):
              wt =   rel_sim * Pr
           elif((state != 0) and (p == 0)):
              wt = 0
           elif((state == 0) and (p == 0)):
              wt = rel_sim
           elif((state == 0) and (p != 0)):
              wt = 0
              
           prob = prob + wt * bn.node[user_nodes[usr]]['prob'][state] 

   bn.node["a_cf"]['prob'][p]  = prob
    

 
###############################  read the user ratings and movie information data ####################
ratings_data =pd.read_csv('F:/Projects/PGM/Movie Recommendation/ml-100k/u_data.csv')

movie_data =pd.read_csv('F:/Projects/PGM/Movie Recommendation/ml-100k/u_items.csv', encoding = "ISO-8859-1")



######################### slpit the ratings data into train and test sets
ratings_train, ratings_test = train_test_split(ratings_data,test_size=0.20, random_state=42)

ratings_train.index = range(len(ratings_train))
ratings_test.index = range(len(ratings_test))


######################### create the user ratings and movie features matrices #######################
movie_feat_mat = movie_data.drop(movie_data.columns[[ 1,2,3]], axis=1)
movie_feat_mat.drop(movie_feat_mat.columns[[0]], axis=1, inplace=True)

user_rat_mat = np.ndarray((num_users,num_movies))

for i in range(0,len(ratings_train)):
  r =ratings_train['user id'][i]
  c = ratings_train['item id'][i]
  user_rat_mat[(r-1),(c-1)] = ratings_train['rating'][i]
user_rat_mat = pd.DataFrame(user_rat_mat)    



########### find the users similar to the active user , based on Pearson's correlation coefficient #################
sim = np.zeros((len(user_rat_mat),len(user_rat_mat)))


for  active_user  in  range(0,len(user_rat_mat)):
    
   item_rat_act_user = np.nonzero(user_rat_mat.ix[active_user,:])[0]
   ra_bar = np.mean(user_rat_mat.ix[active_user,item_rat_act_user])
   
   for usr in range(0,len(user_rat_mat)):
      itms_rat_usr = np.nonzero(user_rat_mat.ix[usr,:])[0]
      rb_bar = np.mean(user_rat_mat.ix[usr,itms_rat_usr])
      rat_both = np.intersect1d(item_rat_act_user,itms_rat_usr)
      num = 0    
      for itm in rat_both:
          num = num + (user_rat_mat.ix[active_user,itm] - ra_bar) * (user_rat_mat.ix[usr,itm] - rb_bar)
        
      den1 = 0
      for itm in rat_both:
          den1 = den1 + (user_rat_mat.ix[active_user,itm] - ra_bar)**2
        
      den2 = 0
      for itm in rat_both:
          den2 = den2 + (user_rat_mat.ix[usr,itm] - rb_bar)**2
      if( (den1 == 0) or (den2 == 0)):
          sim[active_user][usr]=0
      else:
         sim[active_user][usr] = num/math.sqrt(den1*den2)
         sim[active_user][usr] = sim[active_user][usr] * (len(rat_both)/len(item_rat_act_user))
    
###############################################################################################################


############## find the users similar to the active user , based on new heuristic measure #################
sim_nhm = np.zeros((len(user_rat_mat),len(user_rat_mat)))

for  active_user  in  range(0,len(user_rat_mat)):
    
   item_rat_act_user = np.nonzero(user_rat_mat.ix[active_user,:])[0] 
   
   for usr in range(0,len(user_rat_mat)):
      itms_rat_usr = np.nonzero(user_rat_mat.ix[usr,:])[0]
      rb_bar = np.mean(user_rat_mat.ix[usr,itms_rat_usr])
      rat_both = np.intersect1d(item_rat_act_user,itms_rat_usr)
      
      pss = 0    
      for itm in rat_both:
          r_med_ind = np.nonzero(user_rat_mat.ix[:,itm])[0]
          r_med = np.median(user_rat_mat.ix[r_med_ind,itm])
          mu_p = np.mean(user_rat_mat.ix[r_med_ind,itm])
          
          r_u = user_rat_mat.ix[active_user,itm]
          r_v = user_rat_mat.ix[usr,itm]
          r_avg = (r_u + r_v)/2

          Prox = 1 - ( 1 / (1 + math.exp(-abs(r_u - r_v))) ) 
          Sig = 1 / (1 +  math.exp(-abs(r_u-r_med)*abs(r_v-r_med)) )
          Sing = 1 - (1 / (1 + math.exp(-abs(r_avg - mu_p))) )
          
          pss = pss + Prox*Sig*Sing
        
      jac_sim = len(rat_both) / (len(itms_rat_usr) * len(item_rat_act_user))
      jpss = pss * jac_sim
      
      mu_u = np.mean(user_rat_mat.ix[active_user,item_rat_act_user])      
      mu_v = np.mean(user_rat_mat.ix[usr,itms_rat_usr])      
      
      sd_u = np.std(user_rat_mat.ix[active_user,item_rat_act_user])
      sd_v = np.std(user_rat_mat.ix[usr,itms_rat_usr])
      
      urp_sim = 1 - ( 1 / (1+ math.exp(-abs(mu_u-mu_v)*abs(sd_u-sd_v))) )
      
      sim_nhm[active_user][usr] = jpss * urp_sim
      
###############################################################################################################


expected = ratings_test['rating'][0:1000]
predicted = np.zeros(1000)

####### iterate over the test ratings set
for k in range(0,1000):
  
  ################# create random variables for storing pobabilities for features, items and users #######
  feat_prob = np.zeros((num_features,2))
  item_prob= np.zeros((num_movies,2))
  user_prob = np.zeros((num_users,6))
  a_cb_prob =  np.zeros(6)
  a_cf_prob =  np.zeros(6)
  a_h_prob =  np.zeros(6)


  ############### create nodes names  #################################################################
  feat_nodes = ["feat_" + str(i) for i in range(0,num_features)]
  item_nodes = ["item_" + str(i) for i in range(0, num_movies)]
  user_nodes = ["user_" + str(i) for i in range(0, num_users)]



  ############################## set the active user ######################################################
  active_user = (ratings_test['user id'][k]) - 1


  ######################### create the static part of the Bayesian Netwotk for the active user ##############
  bn = nx.DiGraph()


  ########################## add all the feature nodes to the Bayesian network  ############################
  for i in range(0,num_features):
    bn.add_node(feat_nodes[i],prob = feat_prob[i])


  ######################### assign the a priori probabilities to the feature nodes ########################
  for i in range(0,num_features):
    num = np.nonzero(movie_feat_mat.ix[:,i])[0]
    bn.node[feat_nodes[i]]['prob'][1] = len(num) / num_movies
    bn.node[feat_nodes[i]]['prob'][0] = 1 - bn.node[feat_nodes[i]]['prob'][1]


  ####################### add the relevant item nodes , corresponding to the movies rates by the active user
  item_rat_act_user = np.nonzero(user_rat_mat.ix[active_user,:])[0] ## items rated by the active user

  for itm in item_rat_act_user:
    bn.add_node(item_nodes[itm],prob = item_prob[itm])



##################### add edges from the feature nodes to the item nodes ##################################
  for itm in item_rat_act_user:
      parents =  np.nonzero(movie_feat_mat.ix[itm,:])[0]
      for ft in parents:
         bn.add_edge(feat_nodes[ft],item_nodes[itm])



  ################### add a node for A_CB and  edges from all the item nodes to the A_CB node #################
  bn.add_node("a_cb",prob = a_cb_prob)
  for itm in item_rat_act_user:
      bn.add_edge(item_nodes[itm],"a_cb")



#  users_sim_act_user = heapq.nlargest(30, range(len(sim)), sim.take)  ##### select the top 30 similar users 

        
      
  users_sim_act_user_nhm = heapq.nlargest(20, range(len(sim_nhm[active_user])), sim_nhm[active_user].take)  ##### select the top 30 similar users 


  ##################### add the similar nodes to the Bayes Network #############################################
  for usr in users_sim_act_user_nhm:   #### add the similar users to the BN
    bn.add_node(user_nodes[usr],prob = user_prob[usr])


  bn.add_node("a_cf",prob = a_cf_prob)  ###### add a node for A_CF 


  for usr in  users_sim_act_user_nhm:     ####### add an edge from every similar item to A_CF
      bn.add_edge(user_nodes[usr],"a_cf")



########################## add a node for the Hybrid part to collect info from both CB and CF parts  ##########
  bn.add_node("a_h",prob = a_h_prob)

  bn.add_edge("a_cb","a_h")
  bn.add_edge("a_cf","a_h")



  ############################# Expand the BN by adding the target item node #####################################
  target_item = (ratings_test['item id'][k]) - 1
  par_target_item = np.nonzero(movie_feat_mat.ix[target_item,:])[0]

  bn.add_node(item_nodes[target_item],prob = item_prob[target_item])  #### add the targer item

  for feat in par_target_item:                  ##### add edges from the features describing it 
    bn.add_edge(feat_nodes[feat],item_nodes[target_item])    

  new_item_set = np.append(item_rat_act_user,target_item)



  ######################### Expand the BN by adding edges from items to U- #####################################
  u_minus = list() 
 
  for usr in users_sim_act_user_nhm:              ####### identify the user nodes in U-
    if(user_rat_mat.ix[usr,target_item] == 0 ):
          u_minus.append(usr)

  for usr in u_minus:                         ####### add edges from items to the users in U-
    par = np.nonzero(user_rat_mat.ix[usr,:])[0]
    for itm in par:
          bn.add_node(item_nodes[itm],prob = item_prob[itm])
          bn.add_edge(item_nodes[itm],user_nodes[usr])
          if((item_nodes[itm] in bn.nodes()) == False):
              par_feat = np.nonzero(movie_feat_mat.ix[itm,:])[0]
              for feat in par_feat:
                  bn.add_edge(feat_nodes[feat],item_nodes[itm])

        


  ############################ Set the item evidence by instantiating the target item #############################
  bn.node[item_nodes[target_item]]['prob'][1] = 1
  bn.node[item_nodes[target_item]]['prob'][0] = 0



  ############################# Propgate the item evidence to the parent features ############################# 
  par_item_evid = np.nonzero(movie_feat_mat.ix[target_item,:])[0]

  M = 0
  for feat in par_item_evid:
    nk = len(np.nonzero(movie_feat_mat.ix[:,feat])[0])
    M = M + math.log((num_movies/nk) + 1) 
   
  Pr_itm = 0   
  for feat in par_item_evid:
      wt = math.log((num_movies/nk) + 1) * (1/M)
      Pr_itm = Pr_itm + (bn.node[feat_nodes[feat]]['prob'][1]) * wt
   
  for feat in par_item_evid:
      wt = math.log((num_movies/nk) + 1) * (1/M)
      add = ( wt* (bn.node[feat_nodes[feat]]['prob'][1]) * (1 - bn.node[feat_nodes[feat]]['prob'][1]) ) / Pr_itm
      bn.node[feat_nodes[feat]]['prob'][1] = bn.node[feat_nodes[feat]]['prob'][1] + add
      bn.node[feat_nodes[feat]]['prob'][0] = 1 -  bn.node[feat_nodes[feat]]['prob'][1] 




  ############################### set the collaborative user evidence ########################################
  for usr in users_sim_act_user_nhm:
    if( (usr in u_minus) == False ):
        rat = user_rat_mat.ix[usr,target_item]
        for i in range(0,6):
            if(rat == i):
                bn.node[user_nodes[usr]]['prob'][i] = 1
            else:
                bn.node[user_nodes[usr]]['prob'][i] = 0



  ########################### Propagate the evidence from the features to the items ##########################
  for i in range(0,len(item_nodes)):
      if( item_nodes[i] in bn.nodes()):
          item_prob_func(i)

#  for i in range(0,len(item_nodes)):
#      if( item_nodes[i] in bn.nodes()):
#         print( sum(bn.node[item_nodes[i]]['prob']))



  ####################### Propagate evidence to user nodes from items  ######################################
  for usr in u_minus:
    for i in range(0,6):
      user_prob_func(usr,i)

#  for usr in users_sim_act_user:
#      print(sum( bn.node[user_nodes[usr]]['prob']))



  #######################  Propagate evidence to a_cb node from items ######################################
  for i in range(0,6):
      acb_prob_func(i)

#  print(bn.node["a_cb"]['prob'])
        


  ######################## Propogate evidence from users to a_cf ##########################################
  tot_sim = 0
  for i in users_sim_act_user_nhm:
      tot_sim = tot_sim + sim_nhm[active_user][i]

  for i in range(0,6):
      cf_prob_func(i)

#  print(bn.node["a_cf"]['prob'])



  ###################### combining the evidence from the content based and collaborative parts ###########
  alpha = bn.node["a_cf"]['prob'][0]
  
  rating_cb = np.zeros(5)
  for h in range(0,5):
      rating_cb[h] = (bn.node["a_cb"]['prob'][h+1])/(1 - bn.node["a_cb"]['prob'][0])
      
  rating_cf = np.zeros(5)
  for h in range(0,5):
      rating_cf[h] = (bn.node["a_cf"]['prob'][h+1])/(1 - bn.node["a_cf"]['prob'][0])
  
  cb_pred = 0

  if( (rating_cb[0] >= 0.5) and ( np.sum(rating_cb[0:5]) >= 0.5 ) ):
      cb_pred = 1
  elif( (np.sum(rating_cb[0:2]) >= 0.5) and ( np.sum(rating_cb[1:5]) >= 0.5 ) ):
      cb_pred = 2
  elif( (np.sum(rating_cb[0:3]) >= 0.5) and ( np.sum(rating_cb[2:5]) >= 0.5 ) ):
      cb_pred = 3
  elif( (np.sum(rating_cb[0:4]) >= 0.5) and ( np.sum(rating_cb[3:5]) >= 0.5 ) ):
      cb_pred = 4
  else:
      cb_pred = 5 
          
   
  cf_pred = 0
  
  if( (rating_cf[0] >= 0.5) and ( np.sum(rating_cf[0:5]) >= 0.5 ) ):
      cf_pred = 1
  elif( (np.sum(rating_cf[0:2]) >= 0.5) and ( np.sum(rating_cf[1:5]) >= 0.5 ) ):
      cf_pred = 2
  elif( (np.sum(rating_cf[0:3]) >= 0.5) and ( np.sum(rating_cf[2:5]) >= 0.5 ) ):
      cf_pred = 3
  elif( (np.sum(rating_cf[0:4]) >= 0.5) and ( np.sum(rating_cf[3:5]) >= 0.5 ) ):
      cf_pred = 4
  else:
      cf_pred = 5       
  
  
  rating_h = np.zeros(5)
  
  if(cb_pred == cf_pred):
      for i in range(0,5):
         if(i == (cb_pred-1)):      
           rating_h[i] = 1
         else:
           rating_h[i] = 0
  else:
      for i in range(0,5):
          if(i == (cb_pred-1)):
              rating_h[i] = alpha
          elif(i == (cf_pred-1)):
              rating_h[i] = (1-alpha)
          else:
              rating_h[i] = 0
  
 
  
 
  predicted[k] = (np.argmax(rating_h)) + 1  





mad = 0
for i in range(0,1000):
    mad = mad + abs(predicted[i] - expected[i] )

mad = mad/1000



