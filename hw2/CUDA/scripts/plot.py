import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sns.set_theme()

def plot_heatmap(data, n: int):
    fig, ax = plt.subplots(figsize=(10, 10))
    data = data[data['n'] == n]
    data = data.pivot(index='thread_y', columns='thread_x', values='average_time_gpu')
    sns.heatmap(data, annot=True, fmt=".2f", linewidths=0, ax=ax, cmap="vlag")
    ax.set_xlabel('#Thread per block (x)')
    ax.set_ylabel('#Thread per block (y)')
    ax.set_title(f"Heatmap of execution times for n = {n}")
    fig.savefig(f"report/img/heatmap_{n}.pdf")

# def plot_line(data, n: int):
#     data = data[data['n'] == n]
#     fig, ax = plt.subplots(figsize=(10, 10))
#     sns.lineplot(data=data, x="thread_x", y="average_time_gpu", hue="thread_y", ax=ax, marker="o", palette="tab10")
#     ax.set_title(f"Lineplot of execution times for n = {n}")
#     ax.legend(title='#Thread per block (y)')
#     ax.set_xlabel('#Thread per block (x)')
#     ax.set_ylabel('Average time (ms)')
#     fig.savefig(f"report/img/lineplot_{n}.pdf")

def plot_line_3d(data, n: int):
    data = data[data['n'] == n]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(data['thread_x'], data['thread_y'], data['average_time_gpu'], cmap="vlag")
    ax.set_xlabel('#Thread per block (x)')
    ax.set_ylabel('#Thread per block (y)')
    ax.set_zlabel('Average time (ms)')
    ax.invert_yaxis()  # Reverse the order of the y-axis
    ax.set_title(f"3D plot of execution times for n = {n}")
    fig.savefig(f"report/img/lineplot3d_{n}.pdf")

def plot_time(data,n):
    fig, ax = plt.subplots(figsize=(10, 10))
    data = data[data["thread_x"] == data["thread_y"]]
    # for every resolution plot the average time
    data_n = data[data["n"] == n]
    sns.lineplot(
        data=data_n,
        x="thread_x",
        y="average_time_gpu",
        ax=ax,
        marker="o",
        label=f"{n}",
    )
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str((x, x))))
    # plt.yscale("log")
    ax.set_ylabel("Average time (ms)")
    ax.set_xlabel("Thread number")
    ax.set_title(f"Execution times with n={n}")
    fig.savefig(f"report/img/gpu_time_{n}.pdf")

# plot a graph where n = 1000 and the x axis is pair (thread_x, thread_y) and the y axis is the average gpu time for the tuple (thread_x, thread_y)
def plot_cpu_gpu(data, n: int):
    fig, ax = plt.subplots(figsize=(10, 10))
    data = data[data['n'] == n]
    # gropuby and mean to get the average time for each pair where thread_x = thread_y
    data = data[data['thread_x'] == data['thread_y']]
    data = data.groupby(['thread_x', 'thread_y']).mean().reset_index()
    sns.lineplot(data=data, x="thread_x", y="average_time_cpu", ax=ax, marker="o", label="CPU", color="tab:blue")
    ax = sns.lineplot(data=data, x="thread_x", y="average_time_gpu", ax=ax, marker="o", label="GPU", color="tab:orange")
    plt.yscale('log')
    ax.set_xticks([1,2,4,8,16])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str((x,x))))
    ax.set_ylabel("Average time (ms)")
    ax.set_xlabel("Thread number")
    ax.set_title(f"Lineplot of execution times for n = {n}")
    fig.savefig(f"report/img/cpu_gpu_{n}.pdf")

def main():
    data = pd.read_csv('report/data.csv')
    data['average_time_gpu'] = data.groupby(['n', 'thread_x', 'thread_y'])['time_gpu'].transform('mean')
    data['average_time_cpu'] = data.groupby(['n'])['time_cpu'].transform('mean')
    data = data[['n', 'thread_x', 'thread_y', 'average_time_gpu', 'average_time_cpu']].drop_duplicates()
    data.to_csv('report/data_graph.csv', index=False)
    for n in data['n'].unique():
        #plot_heatmap(data, n)
        plot_time(data, n)
        # plot_line_3d(data, n)
        # plot_cpu_gpu(data, n)


if __name__ == "__main__":
    main()
