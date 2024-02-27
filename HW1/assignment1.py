import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp

def line(w,b,x):
    y=-b/w[1]-x*w[0]/w[1]
    y=np.reshape(y,1)
    return y

def plots(X,y,w,b,j,k,savefig):
    xmin=X.min()
    xmax=X.max()
    indexn=np.where(y==-1)
    indexp=np.where(y==1)
    plt.plot(X[0,indexp], X[1,indexp], 'k+')
    plt.plot(X[0,indexn], X[1,indexn], 'k.')
    if (j>=0):
        if (y[j]==1):
            plt.plot(X[0,j], X[1,j], 'ko')
        else:
            plt.plot(X[0,j], X[1,j], 'ko')
    if (k>=0):
        plt.plot([xmin,xmax],[line(w,b,xmin),line(w,b,xmax)],'k')
        if (j==-1):
            plt.title(f"Iteration {k}", fontsize=20)
        else:
            plt.title(f"Iteration {k}, sample "+str(j), fontsize=20)
    plt.axis([-3,3,-3,3])
    plt.gca().set_aspect('equal', adjustable='box')
    if (savefig):
        plt.savefig(f"figure_{k}.png")
        plt.show()
    return

def data(N, using_non_separable_data = False):
    eta=0.2
    if(using_non_separable_data):
        eta = -0.2

    w = np.array([[0.5], [-0.5]])
    b = np.array(0)
    X = np.zeros((2, N))
    y = np.ones((N))


    for var in range(N):
        y_ = -float("inf")
        while((y_ * y[var]) < eta ):
            x = np.random.randn(2,1)
            y_ = w.T@x + b
            y[var] = np.sign(np.random.randn())
        X[:, var] = x.T

    return X, y

def perceptron_rule(using_non_separable_data = False):
    N=100
    eta=0.2
    if(using_non_separable_data):
        eta = -0.2



    w=np.array([[0.5],[-0.5]])
    b=np.array(0)
    X, y = data(N, using_non_separable_data)

    plots(X,y,w,b,-1,-1,True)

    w = np.random.rand(2,1)
    b = np.array(0)
    k = 0
    error_found = True
    max_iter = 20
    while (error_found):
        error_found = False
        for j in range(N):
            x = np.reshape(X[:,j],(2,1))
            y_= np.sign(w.T@x + b)
            if (y_ != y[j] and k < max_iter):
                plots(X,y,w,b,j,k,True)
                k = k + 1
                w = w + x * y[j]
                b = b + y[j]
                error_found = True

    plots(X,y,w,b,-1,k,True)
    R = np.max(sp.distance_matrix(X.T, X.T, p = 2, threshold = 1000000));
    print("Bound: " + str(np.round((R**2 + 1) / eta)))

    return


def minimum_mean_square_error(using_non_separable_data=False):

    Ntrials = 100
    Ntr = 100
    Ntst = 100
    Max_train = 20
    Etr = np.zeros((Ntrials, Max_train - 3))
    Etst = np.zeros((Ntrials, Max_train - 3))


    for i in range(Ntrials):
        Xtr, ytr = data(Ntr, using_non_separable_data)
        Xtst, ytst = data(Ntst, using_non_separable_data)

        Xtr = np.append(Xtr, np.ones((1, Ntr)), axis=0)
        Xtst = np.append(Xtst, np.ones((1, Ntst)), axis=0)

        for j in range(0, Max_train - 3):
            xj = Xtr[:, 0:j+3]
            yj = ytr[0:j+3]

            w = np.linalg.inv(xj@xj.T)@xj@yj
            y_tr = np.sign(xj.T@w)
            Etr[i,j] = np.mean(yj != y_tr)
            y_tst = np.sign(Xtst.T@w)
            Etst[i,j] = np.mean(ytst != y_tst)

    e_tr = np.mean(Etr, axis=0)
    e_tst = np.mean(Etst, axis=0)

    plt.plot(range(3, Max_train), e_tr, label="e_tr")
    plt.plot(range(3, Max_train), e_tst, label="e_tst")

    plt.title(f"MMSE using_non_separable_data={using_non_separable_data}")
    plt.xlabel("number of training samples")
    plt.ylabel("error")
    plt.legend()

    plt.savefig("minimum_mean_square_error.png")
    plt.show()


    return



def least_mean_squares(using_non_separable_data=False):
    Ntrials = 10000
    Ntr = 100


    Etr = np.zeros((Ntrials, Ntr))
    Ptr = np.zeros((Ntrials, Ntr))

    mu = 0.1

    for i in range(Ntrials):
        Xtr, ytr = data(Ntr, using_non_separable_data)
        Xtr = np.append(Xtr, np.ones((1,Ntr)), axis = 0)
        w = np.array([0,0,0]).T

        for j in range(Ntr):
            e = w.T@Xtr[:,j] - ytr[j]
            w = w - mu*Xtr[:,j] * e
            Etr[i,j] = e**2
            Ptr[i,j] = np.sign(w.T@Xtr[:,j]) != ytr[j]

    e_tr = np.mean(Etr, axis=0)
    p_tr = np.mean(Ptr, axis=0)

    plt.plot(range(Ntr), e_tr, label="e_tr")
    plt.plot(range(Ntr), p_tr, label="p_tr")

    plt.title(f"LMS using_non_separable_data={using_non_separable_data}")
    plt.xlabel("number of training samples")
    plt.ylabel("error")
    plt.legend()

    plt.savefig("least_mean_squares.png")

    plt.show()

    return


def main():

    perceptron_rule(using_non_separable_data = False)
    perceptron_rule(using_non_separable_data = True)


    minimum_mean_square_error(using_non_separable_data = False)
    least_mean_squares(using_non_separable_data = False)

    return


if(__name__ == "__main__"):
    main()
