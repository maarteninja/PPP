import matplotlib.pyplot as plt
from scipy import misc

def test(folder='../data/test/'):
    for i in range(1, 4):
        p = misc.imread('%stest%d.jpg' % (folder, i))
        plt.imshow(p)
        plt.show()

if __name__ == '__main__':
    test()
