import subprocess
import time

EXECS = 2
DEBUG = True

def run_command(command):
   start_time = time.time()
   subprocess.run(command, shell=True)
   end_time = time.time()
   return end_time - start_time

def write_metrics(data):
   with open("./data/data.txt", "w") as f:
      for target, times in data.items():
         f.write(f"{target}:\n")
         for entry in times:
            f.write(f"    - N = {entry['N']}:\n")
            f.write(f"        Time = {entry['Time (s)']:.4f} seconds\n")
         f.write("\n")

def main():
   N = [2_000, 4_000, 6_000, 8_000]
   
   make_targets = ["default", "vec", "parallel"]
   executable = "./release/omp_homework"  # Replace with your actual executable name
   data = {target: {} for target in make_targets}  # Data structure to hold N and Time for each target
   
   for target in make_targets:
      for n in N:
         for run in range(EXECS):
            if DEBUG:
               print(f"N={n}: make {target}")
            subprocess.run("make " + target, shell=True)
            exec_time = run_command(f"{executable} {n}")
            # Store execution time under the corresponding target and N
            data[target] = {"N": n, "run": run, "Time (s)": exec_time}

   print(data)
      # write_metrics(data)

if __name__ == "__main__":
   main()
