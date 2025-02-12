import subprocess
import time
import json

EXECS = 2
DEBUG = True

def run_command(command):
   start_time = time.time()
   subprocess.run(command, shell=True)
   end_time = time.time()
   return end_time - start_time

def write_metrics(data):
   with open("./data/data_auto.json", "w") as f:
      json.dump(data, f, indent=4)

def main():
   N = [20000, 40000, 60000, 80000]

   make_targets = ["vec", "parallel"]
   executable = "./release/omp_homework"  # Replace with your actual executable name
   data = {target: [] for target in make_targets}  # Data structure to hold N and Time for each target

   for target in make_targets:
      for n in N:
         for run in range(EXECS):
            if DEBUG:
               print(f"N={n}: make {target}")
            subprocess.run(f"make {target}", shell=True)
            exec_time = run_command(f"{executable} {n}")
            # Store execution time under the corresponding target and N
            data[target].append({
               "N": n,
               "Time (s)": exec_time,
               "GFLOPS": 0,  # Placeholder, replace with actual calculation if needed
               "Hotspots": [
                  {
                     "Hotspot": "-",  # Placeholder, replace with actual hotspot data if needed
                     "Time (s)": 0  # Placeholder, replace with actual hotspot time if needed
                  }
               ]
            })

   write_metrics(data)

if __name__ == "__main__":
   main()
