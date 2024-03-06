import seaborn as sns
import matplotlib.pyplot as plt

def main()->None:
    taxi_data = sns.load_dataset('taxis')


    print(taxi_data.head())
    sns.relplot(x='distance', y='fare', data=taxi_data)
    plt.savefig(f"{abs(int(hash(__file__)))}_0.png")
    plt.show()


    sns.scatterplot(x='distance', y='fare', data=taxi_data, markers=['o', 's'], color="green")
    plt.xlabel('distance')
    plt.ylabel('fair')
    plt.title('distance vs fare')
    plt.grid(True)
    plt.savefig(f"{abs(int(hash(__file__)))}_1.png")
    plt.show()


    sns.histplot(data=taxi_data, x='distance', hue='passengers', multiple="stack")
    plt.xlabel('distance')
    plt.ylabel('number of instances')
    plt.title('Histogram of passengers by distance')
    plt.grid(True)
    plt.savefig(f"{abs(int(hash(__file__)))}_2.png")
    plt.show()
    return


if(__name__ == "__main__"):
    main()
