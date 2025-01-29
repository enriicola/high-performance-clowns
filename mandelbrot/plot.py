import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python plot.py <filename>")
    sys.exit(1)

  filename = sys.argv[1]

  with open(filename, "r") as f:
    lines = f.read().strip().split("\n")
    data = [line.strip().split(',') for line in lines]
    matrix = np.array(data, dtype=float)

  plt.imshow(matrix, cmap='magma_r')
  plt.axis("off")
  plt.savefig("mandelbrot.png")
  plt.show()
