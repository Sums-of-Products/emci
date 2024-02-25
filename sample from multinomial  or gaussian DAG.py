import random
import numpy as np


def generate_combinations(numbers):
    if len(numbers) == 0:
        return [[]]
    
    smaller_combinations = generate_combinations(numbers[1:])
    combinations = []
    for i in range(numbers[0]):
        for smaller_combination in smaller_combinations:
            combinations.append([i] + smaller_combination)
    return combinations


def sample_from_dag_multinomial(DAG,dist_dag,n):
    dag=[]
    orp=[]
    for i in range (len(DAG[0])):
        if np.sum(DAG[:,i])==0:
            orp.append(i)
            #print(orp)
    dag.append(orp)
    #print(dag)
    temp=np.copy(dag)
    temp=list(temp)
    while len(temp)!=0:
        chi=[]
        par=temp.pop()
        for i in (par):
            for j in range(len(DAG[0])):
                if DAG[i,j]==1:
                    chi.append(j)
        if len(chi)>0:
            dag.append(list(set(chi)))
            temp.append(list(set(chi)))
    #print(dag)
    
    for i in range(len(dag)):
        current_list = dag[i]
        for j in range(i + 1, len(dag)):
            other_list = dag[j]
            duplicates = set(current_list) & set(other_list)
            for duplicate in duplicates:
                while duplicate in current_list:
                    current_list.remove(duplicate)
    #print(dag)
    
    
    distributions_shape={}
    for i in range(len(DAG[0])):
        index=tuple(np.where(DAG[:,i]==1)[0])
        #print(index)
        shape=[dist_DAG[i]]
        for j in index:
            shape.append(dist_DAG[j])
            #print('s',shape)
            
        distributions_shape[i]=shape
    #print(distributions_shape)    
    
    distributions={}
    
    for i in range (len(distributions_shape)):
        distributions[i]={}
        comb_list=generate_combinations(distributions_shape[i][1:])
        for j in comb_list:
            distributions[i][tuple(j)]=np.random.dirichlet(np.ones(distributions_shape[i][0]),1)[0]
    
    #print(distributions)
    
    samples=[]
    for t in range(n):
        sample=np.zeros(len(DAG[0]))
        #print(sample)
        for i in dag:
            for j in i:
                parent=np.where(DAG[:,j]==1)
                parent=tuple(parent[0])
                rand=random.random()
                #print(j,rand)
                index=0
                if len(parent)==0:
                    for k in range(len(distributions[j][parent])):
                    #k=0
                        #print(np.cumsum(distributions[j][parent])[k])
                        if rand > np.cumsum(distributions[j][parent])[k]:
                            index+=1
                    sample[j]=index
                else:
                    parent_state=[]
                    for l in parent:
                        parent_state.append(sample[l])
                    parent_state=tuple(parent_state)
                    index=0
                    for k in range(len(distributions[j][parent_state])):
                        if rand > np.cumsum(distributions[j][parent_state])[k]:
                            index+=1
                    sample[j]=index
        samples.append(sample)
                
                        
    return samples


def sample_from_dag_gaussian(DAG,n):
    dag=[]
    orp=[]
    for i in range (len(DAG[0])):
        if np.sum(DAG[:,i])==0:
            orp.append(i)
            #print(orp)
    dag.append(orp)
    #print(dag)
    temp=np.copy(dag)
    temp=list(temp)
    while len(temp)!=0:
        chi=[]
        par=temp.pop()
        for i in (par):
            for j in range(len(DAG[0])):
                if DAG[i,j]==1:
                    chi.append(j)
        if len(chi)>0:
            dag.append(list(set(chi)))
            temp.append(list(set(chi)))
    #print(dag)
    
    for i in range(len(dag)):
        current_list = dag[i]
        for j in range(i + 1, len(dag)):
            other_list = dag[j]
            duplicates = set(current_list) & set(other_list)
            for duplicate in duplicates:
                while duplicate in current_list:
                    current_list.remove(duplicate)
    #print(dag)
    
    
    distributions={}
    #for i in range(len(DAG[0])):
     #   distributions.append((np.random.normal(0,5,1)[0],(1/np.random.gamma(1,1,1)[0])**.5,(np.random.normal(0,5,1)[0]))
    
    tau0=(1/np.random.gamma(1,1,1)[0])**.5
    for i in dag:
            for j in i:
                distributions[j]={}         
                parent=np.where(DAG[:,j]==1)
                parent=tuple(parent[0])
                rand=random.random()
                #print(j,rand)
                index=0
                if len(parent)==0:
                    distributions[j]=(np.random.normal(0,tau0,1)[0],(1/np.random.gamma(1,1,1)[0])**.5)
                else:
                    distributions[j]=[np.random.normal(0,tau0,1)[0],(1/np.random.gamma(1,1,1)[0])**.5]
                    for l in range(len(parent)):
                        distributions[j].append(np.random.normal(0,(1/np.random.gamma(1,1,1)[0])**.5,1)[0])
                    distributions[j].append((1/np.random.gamma(1,1,1)[0])**.5)
                    distributions[j]=tuple(distributions[j])
                                      
                             
                             
                            
        
    samples=[]
    for t in range(n):
        sample=np.zeros(len(DAG[0]))
        #print(sample)
        for i in dag:
            for j in i:
                parent=np.where(DAG[:,j]==1)
                parent=tuple(parent[0])
                rand=random.random()
                #print(j,rand)
                index=0
                if len(parent)==0:
                    sample[j]=np.random.normal(distributions[j][0],distributions[j][1])
                else:
                    parent_state=[]
                    for l in parent:
                        parent_state.append(sample[l])
                    parent_state=tuple(parent_state)
                    b0=np.random.normal(distributions[j][0],distributions[j][1])
                    for k in range(len(parent_state)):
                        b0+=parent_state[k]*distributions[j][k+2]
                    sigma=distributions[j][k+3]
                    sample[j]=np.random.normal(b0,sigma,1)[0]
                    
        samples.append(sample)
    return samples