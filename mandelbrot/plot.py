import numpy as np
import matplotlib.pyplot as plt

def main():
    # Read the CSV file into a NumPy array
    data = np.loadtxt('result.csv', delimiter=',')

    # Plot the Mandelbrot set using imshow
    plt.imshow(data, cmap='viridis', origin='lower', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title('Mandelbrot Set')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == '__main__':
    main()