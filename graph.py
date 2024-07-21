import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Loss graph
    with open('record.txt') as f:
        lines = f.read().split('\n')[:-1]
        g_loss = [float(line.split(' ')[0]) for line in lines]
        d_loss = [float(line.split(' ')[1]) for line in lines]
        x = list(range(len(g_loss)))
        plt.plot(x, g_loss, label='Generator')
        plt.plot(x, d_loss, label='Discriminator')
        plt.legend()
        plt.savefig('graph.jpg')
