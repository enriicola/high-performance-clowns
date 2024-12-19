import subprocess
import time

def run_command(command):
   start_time = time.time()
   subprocess.run(command, shell=True)
   end_time = time.time()
   return end_time - start_time

def main():
   commands = ["make vec", "make"]
   executable = "./release/your_executable"  # Replace with your actual executable name
   metrics = {command: [] for command in commands}
   exec_metrics = {command: [] for command in commands}

   for _ in range(10):
      for command in commands:
         exec_time = run_command(command)
         metrics[command].append(exec_time)
         exec_time = run_command(executable)
         exec_metrics[command].append(exec_time)

   with open("metrics.txt", "w") as f:
      for command, times in metrics.items():
         f.write(f"Command: {command}\n")
         for i, exec_time in enumerate(times):
            f.write(f"Run {i+1}: {exec_time:.4f} seconds\n")
         f.write("\n")
      
      for command, times in exec_metrics.items():
         f.write(f"Executable after {command}\n")
         for i, exec_time in enumerate(times):
            f.write(f"Run {i+1}: {exec_time:.4f} seconds\n")
         f.write("\n")

if __name__ == "__main__":
   main()