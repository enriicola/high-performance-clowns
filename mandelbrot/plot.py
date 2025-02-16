import numpy as np
import matplotlib.pyplot as plt

def main():
    # Read the CSV file into a NumPy array
    data = np.loadtxt('result.csv', delimiter=',')

    # Plot the Mandelbrot set using imshow
    plt.imshow(data, cmap='gray_r', origin='lower', interpolation='nearest')
    plt.axis('off')
    plt.savefig('mandelbrot.png', dpi=300)

if __name__ == '__main__':
    main()