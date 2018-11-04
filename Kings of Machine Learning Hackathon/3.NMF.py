import pandas as pd
import numpy as np
from scipy import linalg
from numpy import dot
import os,sys

#Read all the input files
store={}
for filename in ['hero_data','train9','train1','test9','test1','sample_submission']:
    store[filename]=pd.read_csv('1.RawData/%s.csv'%filename)
    store[filename]['source']=filename

# Define function for matrix decomposition. If 0 as matrix entity, do not fit that value
def nmf(mat, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    X=mat.fillna(0).values
    test=store['train1']
    # Define index for prediction
    pred_index=mat.fillna(0).stack().to_frame('kda_ratio').reset_index()
    pred_index=pred_index['user_id'].astype(str)+'_'+pred_index['hero_id'].astype(str)

    
    eps = 1e-5
    print 'Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter)
    # mask
    mask = np.sign(X)

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    np.random.seed(1234)
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        if i % 1 == 0 or i == 1 or i == max_iter:
            print 'Iteration {}:'.format(i),
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            print 'fit residual', np.round(fit_residual, 4),
            print 'total residual', np.round(curRes, 4),
            if curRes < error_limit or fit_residual < fit_error_limit:
                break
            pred=pd.DataFrame(A.dot(Y)).stack().to_frame('kda_ratio')
            pred.index=pred_index
            test['prediction']=test['id'].map(pred['kda_ratio'])
            test['error']=np.square(test['prediction']-test['kda_ratio'])
            print 'test error', np.sqrt(test['error'].mean())
    return A,Y

mat=pd.concat([store['test9'],store['train9'],store['train1']]).groupby(['user_id','hero_id'])
# Remove outliers
mat=np.clip(mat['kda_ratio'].mean(),2000,6000).unstack()

# NMF Solution for different number of latent factors
# Best Iteration: i=0, Public LB: 550
A,Y=nmf(mat,2,max_iter=10)
pred=pd.DataFrame(A.dot(Y),index=mat.index,columns=mat.columns).stack().to_frame('pred').reset_index()
pred.index=pred['user_id'].astype(str)+'_'+pred['hero_id'].astype(str)
store['test1']['pred0']=store['test1']['id'].map(pred['pred'])
submission=store['test1'].copy()
submission['kda_ratio']=submission['pred0']
submission[['id','kda_ratio']].to_csv('2.Submissions/3.NMF0.csv',index=False)
    
# Get user, iteam means solution
# Public LB Score: 555
a=pd.concat([store['train9'],store['test9']])
a['kda_ratio']=np.clip(a['kda_ratio'],2000,6000)
user_mean1=a.groupby(['user_id'])['kda_ratio'].mean()
hero_mean1=a.groupby(['hero_id'])['kda_ratio'].mean()
user_mean2=a['kda_ratio']-a['hero_id'].map(hero_mean1)
user_mean2=user_mean2.groupby(a['user_id']).mean()
hero_mean2=a['kda_ratio']-a['user_id'].map(user_mean1)
hero_mean2=hero_mean2.groupby(a['hero_id']).mean()

#Score test data using above parameters
c=store['test1'].copy()
c['F1']=c['user_id'].map(user_mean1)
c['F2']=c['user_id'].map(user_mean2)
c['F3']=c['hero_id'].map(hero_mean1)
c['F4']=c['hero_id'].map(hero_mean2)
c['kda_ratio']=(c['F1']+c['F2']+c['F3']+c['F4'])/2
c[['id','kda_ratio']].to_csv('2.Submissions/4.UserHeroMean6.csv',index=False)

#Make enseble of NMF0 and Mean encoding solution
#Public LB: 549
a=pd.read_csv('2.Submissions/3.NMF0.csv')
b=pd.read_csv('2.Submissions/4.UserHeroMean6.csv')
c=pd.merge(a,b,on=['id'])
c['kda_ratio']=0.5*c['kda_ratio_x']+0.5*c['kda_ratio_y']
c[['id','kda_ratio']].to_csv('2.Submissions/5.Ensemble3.csv',index=False)

