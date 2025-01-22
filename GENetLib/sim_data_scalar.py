import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


'''Example data for method scalar_ge and grid_scalar_ge'''

def sim_data_scalar(rho_G, rho_E, dim_G, dim_E, n, dim_E_Sparse = 0, ytype = 'Survival',
                  n_inter = None, linear = True, seed = 0):
    if dim_E_Sparse > dim_E:
        raise ValueError("dim_E_Sparse should be less than dim_E")
    
    def generate_continuous_data(rho, dim, n):
        cov = np.zeros(shape=(dim, dim))
        mean = np.zeros(dim)
        for i in range(dim):
            for j in range(dim):
                cov[i,j] = rho ** (abs(i-j))
        return np.random.multivariate_normal(mean = mean, cov = cov, size = n)
    
    def censor_data(h, n):
        U = np.random.uniform(1,3,size = n)
        MEAN = U * np.exp(h)
        TIME = np.random.exponential(np.exp(h))
        C = np.random.exponential(MEAN)
        print(min(TIME),max(TIME))
        print(min(C),max(C))
        Y_TIME = np.where(TIME > C, C, TIME)
        Y_EVENT = np.where(TIME > C, 0, 1)
        return Y_TIME.reshape(-1,1), Y_EVENT.reshape(-1,1)
    
    def generate_ge(seed, ytype):
        np.random.seed(seed)
        X = generate_continuous_data(rho_G, dim_G, n)
        CLINICAL = generate_continuous_data(rho_E, dim_E, n)
        if dim_E_Sparse != 0:
            CLINICAL[:,dim_E-dim_E_Sparse:dim_E] = np.where(CLINICAL[:,dim_E-dim_E_Sparse:dim_E] > 0, 1, -1)
        INTERACTION = np.zeros(shape=(n, dim_G * dim_E))
        k = 0
        for i in range(dim_E):
            for j in range(dim_G):
                INTERACTION[:,k] = CLINICAL[:,i] * X[:,j]
                k = k + 1      
        if n_inter == None:
            raise ValueError("Please enter n_inter")
        else:
            pos = []
            for i in range(dim_E):
                pos += list(range(dim_G * i, dim_G * i + n_inter))
            interactionPos = np.random.choice(pos, size = n_inter, replace=False)
            
            if ytype == 'Survival':
                if linear == True:
                    coef = np.random.uniform(0.5, 0.8, size = n_inter*2+dim_E)
                    h = np.sum(X[:,0:n_inter] * coef[0:n_inter], axis = 1) + np.sum(INTERACTION[:,interactionPos] * coef[n_inter:n_inter*2], axis = 1) + np.sum(CLINICAL * coef[n_inter*2:n_inter*2+dim_E], axis = 1)
                elif linear == False:
                    h = np.sum(np.sin(X[:,0:n_inter]), axis = 1)+np.sum(np.sin(INTERACTION[:,interactionPos]), axis = 1)+np.sum(np.sin(CLINICAL), axis = 1)
                else:
                    raise ValueError("Please enter True or False")
                Y_TIME, Y_EVENT = censor_data(h, n)
                print('Censor rate is: ',(sum(Y_EVENT)/n)[0])
                X = StandardScaler().fit(X).transform(X)
                CLINICAL = StandardScaler().fit(CLINICAL).transform(CLINICAL)
                INTERACTION = StandardScaler().fit(INTERACTION).transform(INTERACTION)
                DATA = np.hstack((X, INTERACTION, CLINICAL, Y_TIME, Y_EVENT))
                data = pd.DataFrame(DATA)
                data.rename(columns={dim_G+dim_E+dim_G*dim_E:'time',dim_G+dim_E+dim_G*dim_E+1:'event'},inplace = True)
            
            elif ytype == 'Continuous':
                coef = np.random.uniform(0.5, 0.8, size = n_inter*2+dim_E)
                bias = np.random.rand(n).reshape(-1,1)
                Y = (np.sum(X[:,0:n_inter] * coef[0:n_inter], axis = 1) + np.sum(INTERACTION[:,interactionPos] * coef[n_inter:n_inter*2], axis = 1) + np.sum(CLINICAL * coef[n_inter*2:n_inter*2+dim_E], axis = 1)).reshape(-1,1) + bias
                X = StandardScaler().fit(X).transform(X)
                CLINICAL = StandardScaler().fit(CLINICAL).transform(CLINICAL)
                INTERACTION = StandardScaler().fit(INTERACTION).transform(INTERACTION)
                DATA = np.hstack((X, INTERACTION, CLINICAL, Y))
                data = pd.DataFrame(DATA)
                data.rename(columns={dim_G+dim_E+dim_G*dim_E:'y_continuous'},inplace = True)
            
            elif ytype == 'Binary':
                coef = np.random.uniform(0.5, 0.8, size = n_inter*2+dim_E)
                bias = np.random.rand(n).reshape(-1,1)
                Y_ = (np.sum(X[:,0:n_inter] * coef[0:n_inter], axis = 1) + np.sum(INTERACTION[:,interactionPos] * coef[n_inter:n_inter*2], axis = 1) + np.sum(CLINICAL * coef[n_inter*2:n_inter*2+dim_E], axis = 1)).reshape(-1,1) + bias
                Y = (Y_ >= np.mean(Y_)).astype(int)
                X = StandardScaler().fit(X).transform(X)
                CLINICAL = StandardScaler().fit(CLINICAL).transform(CLINICAL)
                INTERACTION = StandardScaler().fit(INTERACTION).transform(INTERACTION)
                DATA = np.hstack((X, INTERACTION, CLINICAL, Y))
                data = pd.DataFrame(DATA)
                data.rename(columns={dim_G+dim_E+dim_G*dim_E:'y_binary'},inplace = True)
            else:
                raise ValueError("Invalid ytype")
        return {'data':data, 'interpos':interactionPos}
    return generate_ge(seed, ytype)

