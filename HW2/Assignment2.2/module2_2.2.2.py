import torch
from torch.utils.data import TensorDataset, DataLoader


class MyPyTorchShuffler:
    def __init__(self, my_input_as_a_list):
        self.my_data = self.set_data(my_input_as_a_list)

    def set_data(self, assign_me):
        return DataLoader(TensorDataset(torch.tensor(assign_me)), batch_size=3, shuffle=True)

    def get_data(self):
        return self.my_data

    def filter_evens(self):
        return [item.item() for batch in self.get_data() for item in batch[0] if item.item() % 2 == 0]

    def square(self):
        return [((item.item())**2) for batch in self.get_data() for item in batch[0] if item.item()]



def main()->None:
    raw_data = [4, 16, 7, 12, 32, 10, 2, 9, 14, 24, 30, 20, 8, 15]


    new_filter = MyPyTorchShuffler(raw_data)

    new_filtered_list_evens = new_filter.filter_evens()
    new_squared_list = new_filter.square()
    print(f"list using pytorch\t\t {new_filtered_list_evens},{new_squared_list}")

    new_filtered_list_evens.sort()
    new_squared_list.sort()

    print(f"sorted using pytorch\t\t {new_filtered_list_evens},{new_squared_list}")
    return


if(__name__ == "__main__"):
    main()
