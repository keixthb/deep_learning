import matplotlib.pyplot as plt
import numpy as np
import os

import numpy as np

dataset='circle'
output_activation="logistic"

hidden_activation="relu"

D=np.array([2,100,1])


s=0.05
Ntrain=500
Ntest=1000
np.random.seed(30)

def data(name,K,sigma, inner_radius=0.8, outer_radius=1.0):
    if name=='xor' :
        N=round(K/4)
        classes=[0,1,1,0]
        X=sigma*np.random.randn(2,4*N)
        mean=np.array([[-1,-1, 1, 1],[-1,1,-1,1]])
        M=np.ones((N,2))*mean[:,0]
        y=np.ones((1,N))*classes[0]
        for i in range(1,4):
            m=np.ones((N,2))*mean[:,i]
            M=np.concatenate((M,m))
            y=np.concatenate((y,np.ones((1,N))*classes[i]),axis=1)
        M=M.T
        X=X+M

    elif name=='circle':
        N=K;
        theta=np.random.rand(1,N)*np.pi*2
        rho=np.random.randn(1,N)*sigma+inner_radius
        X1=rho*np.block([[np.cos(theta)],[np.sin(theta)]])

        theta=np.random.rand(1,N)*2*np.pi
        rho=np.random.randn(1,N)*sigma+outer_radius
        X2=rho*np.block([[np.cos(theta)],[np.sin(theta)]])

        y=np.concatenate((0*np.ones((1,N)),np.ones((1,N))),axis=1)
        X=np.concatenate((X1,X2),axis=1)

    return X,y


def sigma(z,activation):
    if activation=="logistic":
        h=1/(1+np.exp(-z))
        dh=h*(1-h)
    elif activation=="linear":
        h=z
        dh=np.ones((1,z.size))
    elif activation=="relu":
        h=np.maximum(0,z)
        dh=np.maximum(0,z>0)
    elif activation=="maxout":
        pass
    elif activation=="softmax":
        pass
    return h,dh

def layer(Di,Do,activation):
    W=np.random.randn(Di,Do)/np.sqrt(Di)
    b=np.random.randn(1,Do)/np.sqrt(Di)
    return W,b,activation

def neural_network(D,activations,output):
    NN={"weights":[],"bias":[],"activation":[]}
    for i in range(D.size-1):
        W,b,activation =  layer(D[i],D[i+1],activations)
        NN["weights"].append(W)
        NN["bias"].append(b)
        NN["dimensions"]=D
        if i<D.size-2:
            NN["activation"].append(activation)
        else:
            NN["activation"].append(output)
    return NN

def forward(NN,X):
    N=X.shape[-1]
    aux={"N":N,"h":[],"dh":[],"z":[]}
    D=NN["dimensions"]
    for i in range(D.size-1):
        aux["h"].append(np.zeros((NN["weights"][i].shape[1],N)))
        aux["dh"].append(np.zeros(( NN["weights"][i].shape[1],N)))
        aux["z"].append(np.zeros(( NN["weights"][i].shape[1],N)))
    NN.update(aux)

    NN["z"][0]=NN["weights"][0].T@X+NN["bias"][0].T
    NN["h"][0],NN["dh"][0]=sigma(NN["z"][0],NN["activation"][0])

    for i in range(D.size-2):
        NN["z"][i+1]=NN["weights"][i+1].T@NN["h"][i]+NN["bias"][i+1].T
        NN["h"][i+1],NN["dh"][i+1]=sigma(NN["z"][i+1],NN["activation"][i+1])
    return NN


def backward(NN,X,y,mu,gamma):
    D=NN["dimensions"]
    N=X.shape[-1]
    if NN["activation"][-1]=="linear":
        dJ=NN["h"][-1]-y
        delta = dJ*NN["dh"][-1]

    if NN["activation"][-1]=="logistic":
        o,do=sigma((2*y-1)*NN["z"][-1],"logistic")
        delta = (1-2*y)*(1-o)

    if NN["activation"][-1]=="softmax":
        pass

    NN["weights"][-1]=(1-gamma)*NN["weights"][-1]-mu*NN["h"][-2]@delta.T/N
    NN["bias"][-1]=(1-gamma)*NN["bias"][-1]-mu*np.sum(delta)/N
    for i in range(D.size-2):
        j=-i-1
        delta=NN["dh"][j]*delta
        NN["weights"][j]=(1-gamma)*NN["weights"][j]-mu*NN["h"][j-1]@delta.T/N
        NN["bias"][j]=(1-gamma)*NN["bias"][j]-mu*np.sum(delta)/N
        delta=NN['weights'][j]@delta
    return NN

def mse(NN,y):
    e=np.sum(np.square(NN["h"][-1]-y),1)/y.shape[1]
    return e




def compute_circle(radius, index):
    X,y=data(dataset,Ntrain,s, inner_radius=radius, outer_radius=1)


    np.random.seed(40)
    Xtst,ytst=data(dataset,Ntrain,s, inner_radius=radius, outer_radius=1)


    NN=neural_network(np.array(D),hidden_activation,output_activation)

    epochs=1000
    E=np.zeros((epochs))
    Etst=np.zeros((epochs))

    for i in range(epochs):
        NN=forward(NN,X)
        NN=backward(NN,X,y,0.5,0.000)
        E[i]=mse(NN,y)

        NN=forward(NN,Xtst)
        Etst[i]=mse(NN,ytst)


    minx=np.ndarray.min(X)
    maxx=np.ndarray.max(X)
    x1=np.linspace(minx,maxx,100)
    X1=np.tile(x1,(100,1))
    X2=X1.T
    xtst=np.array([X2.reshape(10000), X1.reshape(10000)]);
    NN=forward(NN,xtst)
    output=NN["h"][-1]
    O=output.reshape(100,100);
    [label,indexn]=np.where(ytst==0)
    [label,indexp]=np.where(ytst==1)
    plt.plot(Xtst[0,indexp], Xtst[1,indexp], 'k.')
    plt.plot(Xtst[0,indexn], Xtst[1,indexn], 'k+')
    plt.contour(x1,x1,O,[0.5], colors="red")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"$\\rho_i$={radius}")

    plt.savefig(f"fig_{index}.png")

    plt.show()
    plt.cla()
    return


def main()->None:
    compute_circle(1.1, 0)

    return

if("__main__" == __name__):
    main()
