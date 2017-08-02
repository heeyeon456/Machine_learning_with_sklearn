
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
from numpy.random import RandomState

from tools import rmse_cal,mae_cal,cor_cal,mean_cal,frange,accuracy,precision,recall,aupr,\
		f1_score,make_binary	

def regression_cv(n,model,data):
    
    pred=pd.DataFrame()
    real=pd.DataFrame()
    
    data_t=data.iloc[:,1:].transpose()
    
    kf=KFold(n_splits=n,shuffle=True)
    prediction=pd.DataFrame(columns=['real','pred'])
        
    for train, test in kf.split(data_t):
        
        x_train=data_t.iloc[train,:-1].astype('float64').values
        y_train=data_t.iloc[train,-1].astype('float64').values
        model.fit(x_train,y_train)
    
        x_test=data_t.iloc[test,:-1].astype('float64').values
        y_test=data_t.iloc[test,-1].astype('float64').values        
         
        pred=pred.append(pd.DataFrame(model.predict(x_test)))
        real=real.append(pd.DataFrame(y_test))
                
    prediction=pd.concat([real,pred],axis=1)

    rmse=rmse_cal(prediction.iloc[:,0],prediction.iloc[:,1])
    mad=mae_cal(prediction.iloc[:,0],prediction.iloc[:,1])
    cor=cor_cal(prediction.iloc[:,0],prediction.iloc[:,1])
    
    print('rmse : '+str(rmse)+'\nmad : '+str(mad)+'\ncor : '+str(cor[0]))
    
    return prediction

def classification_cv(n,model,data):
    
    pred=pd.DataFrame()
    real=pd.DataFrame()
    
    data_t=data.iloc[:,1:].transpose()
    
    kf=KFold(n_splits=n,shuffle=True)
    prediction=pd.DataFrame(columns=['real','pred'])
    
    y_data=data_t.iloc[:,-1]
    y_data=pd.DataFrame(data=make_binary('normal','cancer',y_data))
    
    
    for train, test in kf.split(data_t):
        
        x_train=data_t.iloc[train,:-1].astype('float64').values
        y_train=y_data.iloc[train,-1].values
        model.fit(x_train,y_train)
    
        x_test=data_t.iloc[test,:-1].astype('float64').values
        y_test=y_data.iloc[test,-1].values        
         
        pred=pred.append(pd.DataFrame(model.predict(x_test)))
        real=real.append(pd.DataFrame(y_test))
                
    prediction=pd.concat([pred,real],axis=1)

    acc=accuracy(prediction.iloc[:,0],prediction.iloc[:,1])
    prec=precision(prediction.iloc[:,0],prediction.iloc[:,1])
    rec=recall(prediction.iloc[:,0],prediction.iloc[:,1])
    f1=f1_score(prediction.iloc[:,0],prediction.iloc[:,1])
    
    print('accuracy : '+str(acc)+'\nprecision :'+str(prec)
         +'\nrecall : '+str(rec)+'\nf1_score : '+str(f1))
    
    return prediction

def test_preprocessing(test_df,dataset): 
    
    if 'CpG_site' in test_df.columns:
        test_df.rename(columns={'CpG_site': 'Composite Element REF'}, inplace=True)
    elif 'ID_REF' in test_df.columns:
        test_df.rename(columns={'ID_REF': 'Composite Element REF'}, inplace=True)
    
    selected=pd.DataFrame(columns=['Composite Element REF'],data=dataset.iloc[:,0])
    raw_data=pd.merge(test_df,selected,how='right',on='Composite Element REF')

    raw_data=raw_data.replace('null',float('nan'))
    raw_data=raw_data.T.fillna(raw_data.mean(axis=1)).T
    raw_data.fillna(0.5,inplace=True)
    test_data=raw_data.transpose()
    
    return test_data

def external_val_reg(testdf,dataset,model):
    
    prediction=pd.DataFrame(columns=['predict','real'])
    
    data_t=dataset.iloc[:,1:].transpose()
    X_data = data_t.iloc[:,:-1].values
    y_data = data_t.iloc[:,-1].values
    
    test_data=test_preprocessing(testdf,dataset)
    test_x=test_data.iloc[1:, :-1].astype('float64').values
    test_y=test_data.iloc[1:, -1].astype('float64').values
    
    model.fit(X_data,y_data)
    
    a=pd.DataFrame(model.predict(test_x))
    real=pd.DataFrame(test_y)
    prediction=pd.concat([a,real],axis=1)

    rmse=rmse_cal(prediction.iloc[:,0],prediction.iloc[:,1])
    mae=mae_cal(prediction.iloc[:,0],prediction.iloc[:,1])
    cor=cor_cal(prediction.iloc[:,0],prediction.iloc[:,1])
    
    print('rmse = '+str(rmse) +'\nmae = '+str(mae)+'\ncor = '+str(cor[0]) )
    
    return prediction


def external_val_classif(testdf,dataset,model):
    
    prediction=pd.DataFrame(columns=['predict','real'])
    
    data_t=dataset.iloc[:,1:].transpose()
    X_data = data_t.iloc[:,:-1].astype('float64').values
    y_data = data_t.iloc[:,-1]
    y_data=make_binary('normal','cancer',y_data).values

    tmp=dataset.iloc[:-1,:]
    test_data=test_preprocessing(testdf,tmp)
    test_x=test_data.iloc[1:, :].astype('float64').values
    test_y=testdf.iloc[-1, 1:]
    test_y=make_binary('normal','cancer',test_y).values
    
    model.fit(X_data,y_data)
    
    a=pd.DataFrame(model.predict(test_x))
    real=pd.DataFrame(test_y)
    prediction=pd.concat([a,real],axis=1)

    acc=accuracy(prediction.iloc[:,0],prediction.iloc[:,1])
    prec=precision(prediction.iloc[:,0],prediction.iloc[:,1])
    rec=recall(prediction.iloc[:,0],prediction.iloc[:,1])
    f1=f1_score(prediction.iloc[:,0],prediction.iloc[:,1])
    
    print('accuracy : '+str(acc)+'\nprecision :'+str(prec)
         +'\nrecall : '+str(rec)+'\nf1_score : '+str(f1))
    
    return prediction

def cal_external_auc(test_df,y_score):

    test_y=test_df.iloc[-1, 1:]
    test_y=make_binary('normal','cancer',test_y).values
    fpr,tpr,threshold = roc_curve(test_y,y_score)
    roc_auc=auc(fpr,tpr)
    aupr_value=aupr(test_y,y_score)
    print('auc : '+roc_auc+'\naupr : '+aupr_value)

    return roc_auc, aupr_value


def cal_auc(inputdf,model,testratio):
    #preprocessing for ROC curve

    input_data=inputdf.iloc[:,1:].transpose()
    X_data=input_data.iloc[:,:-1].values
    y_data=input_data.iloc[:,-1]
    y_data=make_binary('normal','cancer',y_data)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=testratio
                                                    ,random_state=RandomState(None))
    model.fit(X_train,y_train)
    
    y_score=model.decision_function(X_test)
    fpr,tpr,threshold = roc_curve(y_test,y_score,pos_label=1)
    roc_auc=auc(fpr,tpr)  
    Aupr=aupr(y_test,y_score)

    return y_score,fpr,tpr,threshold,roc_auc

def draw_roc(inputdf,model,testratio):

    plt.figure()
    lw=2
    y_score,fpr,tpr,threshold,roc_auc= cal_auc(inputdf,model,testratio)

    plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve (area=%0.2f)' % roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

