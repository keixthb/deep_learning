import tensorflow as tf
from tensorflow.keras import layers, models


class MyTensorFlowShuffler:
    def __init__(self, my_input_as_a_list):
        self.my_data = self.set_data(my_input_as_a_list)

    def get_data(self):
        return self.my_data

    def set_data(self, assign_me):
        return tf.data.Dataset.from_tensor_slices(assign_me)

    def filter_evens(self):
        return [item.numpy() for item in self.get_data().shuffle(buffer_size=3).filter(lambda x: tf.equal(tf.math.mod(x, 2), 0))]

    def square(self):
        return [((item.numpy())**2) for item in self.get_data().shuffle(buffer_size=3)]



def main()->None:
    raw_data = [4, 16, 7, 12, 32, 10, 2, 9, 14, 24, 30, 20, 8, 15]


    my_filter = MyTensorFlowShuffler(raw_data)

    filtered_list_evens = my_filter.filter_evens()
    squared_list = my_filter.square()
    print(f"lists using tensorflow\t\t {filtered_list_evens},{squared_list}")

    filtered_list_evens.sort()
    squared_list.sort()

    print(f"sorted using tensorflow\t\t {filtered_list_evens},{squared_list}")
    return


if(__name__ == "__main__"):
    main()
