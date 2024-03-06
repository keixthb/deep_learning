import matplotlib.pyplot as plt
import sklearn.model_selection
import keras.optimizers

import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(30)

def my_relu(x):
    return tf.maximum(0.0,x)

class Data:
    def __init__(self, N, sigma):
        self.N=N
        self.sigma=sigma

    def data_xor(self, classes):
        X=self.sigma*np.random.randn(2,4*self.N)
        mean=np.array([[-1,-1, 1, 1],[-1,1,-1,1]])
        M=np.ones((self.N,2))*mean[:,0]
        y=np.ones((1,self.N))*classes[0]
        for i in range(1,4):
            m=np.ones((self.N,2))*mean[:,i]
            M=np.concatenate((M,m))
            y=np.concatenate((y,np.ones((1,self.N))*classes[i]),axis=1)
        M=M.T
        X=X+M
        return X,y

def main()->None:
    N=100; sigma=0.6
    classes=[0,1,1,0]
    T=Data(N,sigma)
    X,y=T.data_xor(classes)
    X=np.transpose(X)
    y=np.transpose(y)
    y=np.ravel(y)
    print(y.shape)
    print(X.shape)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
    print(X_train.shape, X_test.shape)


    keras.utils.get_custom_objects().update({'my_relu': my_relu})
    model = keras.models.Sequential()


    model.add(keras.layers.Dense(512, activation=my_relu, input_dim=2))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()


    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=4)

    sns.set()
    acc = hist.history['accuracy']
    val = hist.history['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, '-', label='Training accuracy')
    plt.plot(epochs, val, ':', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(f"{abs(int(hash(__file__)))}_0.png")
    plt.show()



    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)


    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    np.shape(x_train)
    plt.imshow(x_train[1,:,:])
    plt.savefig(f"{abs(int(hash(__file__)))}_1.png")
    plt.show()
    print(y_test[15])

    x_train = x_train.reshape((60000, 28 * 28))
    train_X = x_train.astype('float64') / 255
    x_test = x_test.reshape((10000, 28 * 28))
    test_X = x_test.astype('float64') / 255
    train_y = keras.utils.to_categorical(y_train)
    test_y = keras.utils.to_categorical(y_test)


    model = keras.models.Sequential()
    model.add(keras.layers.Dense(512, activation=my_relu, input_shape=(28 * 28,)))
    model.add(keras.layers.Dense(10, activation='softmax'))


    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_X, train_y, epochs=50, batch_size=32, validation_data=(test_X, test_y))

    plt.figure(figsize=(10,10))

    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')

    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.savefig(f"{abs(int(hash(__file__)))}_2.png")
    plt.show()

    loss,acc = model.evaluate(test_X, test_y, verbose=0)
    print('Accuracy %.3f' % (acc * 100.0))


    return


if(__name__ == "__main__"):
    main()
