import matplotlib.pyplot as plt
import numpy as np




class MyMultilayerPerceptronForTheCircleProblem:
    def __init__(self, inner_radius, outer_radius):
        self.__my_inner_radius = inner_radius
        self.__my_outer_radius = outer_radius
        self.__epochs = 1000
        self.__s = 0.05
        self.__Ntrain = 500
        self.__Ntest = 1000
        self.__D = np.array([2,100,1])
        self.__dataset='circle'
        self.__output_activation="logistic"
        self.__hidden_activation="relu"
        self.__my_results = []


    def __data(self, name, K, sigma, inner_radius = 0.8, outer_radius = 1.0):
        if(name) == 'xor':
            N = round(K / 4)
            classes = [0, 1, 1, 0]
            X = sigma * np.random.randn(2, 4 * N)
            mean = np.array([[-1, -1, 1, 1],[-1, 1, -1, 1]])
            M = np.ones((N, 2)) * mean[:, 0]
            y = np.ones((1, N)) * classes[0]
            for i in range(1, 4):
                m = np.ones((N, 2)) * mean[:, i]
                M = np.concatenate((M, m))
                y = np.concatenate((y, np.ones((1, N)) * classes[i]), axis=1)
            M = M.T
            X = X+M

        elif(name) == 'circle':
            N = K;
            theta = np.random.rand(1, N) * np.pi * 2
            rho = np.random.randn(1, N) * sigma + inner_radius
            X1 = rho * np.block([[np.cos(theta)], [np.sin(theta)]])

            theta = np.random.rand(1, N) * 2 * np.pi
            rho = np.random.randn(1, N) * sigma + outer_radius
            X2 = rho * np.block([[np.cos(theta)], [np.sin(theta)]])

            y = np.concatenate((0 * np.ones((1, N)), np.ones((1, N))), axis=1)
            X = np.concatenate((X1, X2), axis=1)

        return X,y

    def __sigma(self, z, activation):
        if(activation) == "logistic":
            h = 1 / (1 + np.exp(-z))
            dh = h * (1 - h)
        elif(activation) == "linear":
            h = z
            dh = np.ones((1, z.size))
        elif(activation) == "relu":
            h = np.maximum(0, z)
            dh = np.maximum(0, z>0)
        elif(activation) == "maxout":
            pass
        elif(activation) == "softmax":
            pass
        return h, dh

    def __layer(self, Di, Do, activation):
        W = np.random.randn(Di,Do)/np.sqrt(Di)
        b = np.random.randn(1,Do)/np.sqrt(Di)
        return W,b,activation

    def __neural_network(self, D, activations, output):
        NN={"weights":[],"bias":[],"activation":[]}
        for i in range(D.size-1):
            W,b,activation =  self.__layer(D[i],D[i+1],activations)
            NN["weights"].append(W)
            NN["bias"].append(b)
            NN["dimensions"]=D
            if i<D.size-2:
                NN["activation"].append(activation)
            else:
                NN["activation"].append(output)
        return NN

    def __forward(self, NN, X):
        N=X.shape[-1]
        aux={"N":N,"h":[],"dh":[],"z":[]}
        D=NN["dimensions"]
        for i in range(D.size-1):
            aux["h"].append(np.zeros((NN["weights"][i].shape[1],N)))
            aux["dh"].append(np.zeros(( NN["weights"][i].shape[1],N)))
            aux["z"].append(np.zeros(( NN["weights"][i].shape[1],N)))
        NN.update(aux)

        NN["z"][0]=NN["weights"][0].T@X+NN["bias"][0].T
        NN["h"][0],NN["dh"][0] = self.__sigma(NN["z"][0],NN["activation"][0])

        for i in range(D.size-2):
            NN["z"][i+1]=NN["weights"][i+1].T@NN["h"][i]+NN["bias"][i+1].T
            NN["h"][i+1],NN["dh"][i+1] = self.__sigma(NN["z"][i+1],NN["activation"][i+1])
        return NN


    def __backward(self, NN, X, y, mu, gamma):
        D=NN["dimensions"]
        N=X.shape[-1]
        if NN["activation"][-1]=="linear":
            dJ=NN["h"][-1]-y
            delta = dJ*NN["dh"][-1]

        if NN["activation"][-1]=="logistic":
            o,do = self.__sigma((2*y-1)*NN["z"][-1],"logistic")
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

    def __mse(self, NN, y):
        e=np.sum(np.square(NN["h"][-1]-y),1)/y.shape[1]
        return e


    def fit(self):
        X, y = self.__data(self.__dataset, self.__Ntrain, self.__s, inner_radius=self.__my_inner_radius, outer_radius=self.__my_outer_radius)
        Xtst, ytst = self.__data(self.__dataset, self.__Ntrain, self.__s, inner_radius=self.__my_inner_radius, outer_radius=self.__my_outer_radius)
        NN = self.__neural_network(np.array(self.__D), self.__hidden_activation, self.__output_activation)
        E = np.zeros((self.__epochs))
        Etst = np.zeros((self.__epochs))


        for i in range(self.__epochs):
            NN = self.__forward(NN, X)
            NN = self.__backward(NN, X, y, 0.5, 0.000)
            E[i] = self.__mse(NN, y)

            NN = self.__forward(NN, Xtst)
            Etst[i] = self.__mse(NN,ytst)


        minx = np.ndarray.min(X)
        maxx = np.ndarray.max(X)
        x1 = np.linspace(minx, maxx, 100)
        X1 = np.tile(x1, (100,1))
        X2 = X1.T
        xtst = np.array([X2.reshape(10000), X1.reshape(10000)]);
        NN = self.__forward(NN,xtst)
        output = NN["h"][-1]
        O = output.reshape(100, 100);
        [label,indexn] = np.where(ytst == 0)
        [label,indexp]=np.where(ytst == 1)


        self.__my_results = [Xtst, x1, O, indexp, indexn]
        return

    def show(self, save_as_filename=""):
        plt.plot(self.__my_results[0][0,self.__my_results[3]], self.__my_results[0][1,self.__my_results[3]], 'k.')
        plt.plot(self.__my_results[0][0,self.__my_results[4]], self.__my_results[0][1,self.__my_results[4]], 'k+')
        plt.contour(self.__my_results[1], self.__my_results[1], self.__my_results[2], [0.5], colors="red")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"$\\rho_i$=[{self.__my_outer_radius},{self.__my_inner_radius}]")

        if(not save_as_filename):
            save_as_filename = "0"

        plt.savefig(f"{abs(int(hash(__file__)))}_{save_as_filename}.png")

        plt.show()
        plt.cla()
        return

    def __str__(self):
        my_string:str = f"self.__my_inner_radius = {self.__my_inner_radius} \n"
        my_string += f"self.__my_outer_radius = {self.__my_outer_radius} \n"
        my_string += f"self.__epochs = {self.__epochs} \n"
        my_string += f"self.__s = {self.__s} \n"
        my_string += f"self.__Ntrain = {self.__Ntrain} \n"
        my_string += f"self.__Ntest = {self.__Ntest} \n"
        my_string += f"self.__D = {self.__D} \n"
        my_string += f"self.__dataset = {self.__dataset} \n"
        my_string += f"self.__output_activation = {self.__output_activation} \n"
        my_string += f"self.__hidden_activation = {self.__hidden_activation} \n"
        my_string += f"self.__my_results = {self.__my_results} \n"
        return my_string


def main()->None:

    np.random.seed(30)

    my_multilayer_perceptron_for_the_circle_problem = MyMultilayerPerceptronForTheCircleProblem(inner_radius=0.8, outer_radius=1.1)
    my_multilayer_perceptron_for_the_circle_problem.fit()
    my_multilayer_perceptron_for_the_circle_problem.show()

    print(my_multilayer_perceptron_for_the_circle_problem)

    return

if("__main__" == __name__):
    main()
