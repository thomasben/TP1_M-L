import matplotlib.pyplot as plt
import numpy as np

## GENERE LES VARIABLES ARTIFICIELLES

def gen_linear(a=np.array([1,1,1]),b=5,d=3,eps=1,nbex=30):
    X=np.zeros([nbex,d])
    Y=np.zeros([nbex,1])
    epsi=np.random.normal(0,eps,nbex)
    for i in range(nbex):
        X[i,:]=np.random.uniform(-5,5,d)
        Y[i,0]=np.dot(a,X[i,:].T)+b+epsi[i]
    return [X,Y]

TEST=gen_linear(7,1,1,1,100)

#plt.scatter(TEST[0],TEST[1])

## IMPORTE LES DONNEES REELLES

def read_file ( fn ):
    with open ( fn ) as f :
        names = f . readline ()
        X = np . array ([[ float ( x ) for x in l . strip (). split (" ")] for l in f . readlines ()])
    return X [: ,: -1] , X [: , -1]. reshape ( -1)

u=read_file('/home/hugo/Bureau/Machine_learning/housing.csv')
print(u)

def affiche2D():
    plt.figure ()
    plt . subplot (3 ,1 ,1)
    plt . scatter (u[0][:,1],u[0][:,3])
    plt . subplot (3 ,1 ,2)
    plt . scatter (u[0][:,2],u[0][:,4])
    plt . subplot (3 ,1 ,3)
    plt . scatter (u[0][:,3],u[0][:,5])

affiche2D()
plt.show()    
## FAIT LA REGRESSION

def predict(w,X):
    Xp=np.concatenate((np.ones([X.shape[0],1]),X),axis=1)
    return np.dot(Xp,w)

def mse(yhat,y):
    return np.vdot((y-yhat),(y-yhat).T)

def regress(X,Y):
    Xp=np.concatenate((np.ones([X.shape[0],1]),X),axis=1)
    return np.dot(np.linalg.pinv(np.dot(Xp.T,Xp)),np.dot(Xp.T,Y))

def question221(nb_points,pas,t_initial,bruit,bruit_initial):
    table_donnees0=np.zeros(nb_points)
    table_donnees1=np.zeros(nb_points)    
    table_bruit0=np.zeros(nb_points)
    table_bruit1=np.zeros(nb_points)

    for i in range(nb_points):
        res=gen_linear(nbex=pas*i+t_initial)
        w=regress(res[0],res[1])
        table_donnees0[i]=np.mean(mse(predict(w,res[0]),res[1]))
        table_donnees1[i]=mse(w[1:],np.array([1,1,1]))

    for j in range(nb_points):
        res=gen_linear(nbex=200,eps=bruit_initial+j*bruit/nb_points)
        w=regress(res[0],res[1])
        table_bruit0[j]=np.mean(mse(predict(w,res[0]),res[1]))
        table_bruit1[j]=mse(w[1:],np.array([1,1,1]))

    plt.figure()
    plt.subplot (2 ,2 ,1)
    plt.scatter(np.array([(t_initial+i*pas) for i in range (nb_points)]),table_donnees0)
    plt.subplot (2 ,2 ,2)
    plt.scatter(np.array([(bruit_initial+j*bruit/nb_points) for j in range (nb_points)]),table_bruit0)
    plt.subplot (2 ,2 ,3)
    plt.scatter(np.array([(t_initial+i*pas) for i in range (nb_points)]),table_donnees1)
    plt.subplot (2 ,2 ,4)
    plt.scatter(np.array([(bruit_initial+j*bruit/nb_points) for j in range (nb_points)]),table_bruit1)


