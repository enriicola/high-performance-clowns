import subprocess
import itertools as it
import logging
import csv

N = [1000, 10_000, 30_000]
THREADS_X = [1,2,4,8,16]
THREADS_Y = [1,2,4,8,16]
RUNS = 3

COMP = "nvc++"
CFLAGS = []

def compile(in_file_path: str, out_file_path: str, n: int, thread_x: int, thread_y: int):

    subprocess.run(
        [
            COMP,
            *CFLAGS,
            in_file_path,
            "-o",
            "bin/" + out_file_path,
            f"-DNI={n}",
            f"-DNJ={n}",
            f"-DTHREADS_X={thread_x}",
            f"-DTHREADS_Y={thread_y}",
        ]
    )

    res = subprocess.run([f"./bin/{out_file_path}"], stdout=subprocess.PIPE)
    timing_cpu = float(res.stdout.decode("utf-8").split("\n")[0].split(" ")[-3])
    timing_gpu = float(res.stdout.decode("utf-8").split("\n")[1].split(" ")[-3])
    print(f"CPU: {timing_cpu} GPU: {timing_gpu}")
    return (timing_cpu, timing_gpu)

def main():
    with open('report/data.csv', 'w',  newline='') as csvfile:
        fieldnames = ['run', 'n', 'thread_x', 'thread_y', 'time_cpu','time_gpu']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for n, thread_x, thread_y in it.product(N, THREADS_X, THREADS_Y):
            print(n, thread_x, thread_y)
            for run in range(1, RUNS + 1):
                curr_time_cpu, curr_time_gpu = compile("src/heat.cu", f"main_{n}_{thread_x}_{thread_y}", n, thread_x, thread_y)
                writer.writerow({'run': run, 'n':n, 'thread_x':thread_x, 'thread_y':thread_y, 'time_cpu':curr_time_cpu, 'time_gpu': curr_time_gpu})
               
if __name__ == "__main__":
    main()
