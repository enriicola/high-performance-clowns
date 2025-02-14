# Introduction

In this report, we aim to accelerate a given program that computes the 2D heat conduction 
formula using CUDA. The primary objective is to evaluate different configurations of 
blocks and threads, as well as various problem sizes, to achieve optimal performance. We 
will conduct a thorough analysis of the code to identify hotspots and discuss potential 
vectorization issues. The best sequential time will be established as a reference point, 
and performance metrics will be presented using Google Colab. The report will include 
charts to illustrate speedup values, focusing on data sizes that result in 
sequential execution times of 30 seconds or more. Finally, we will draw conclusions based 
on our findings.

# Hardware Capability ⚙️

The VM of google colab is equipped with this CPU:

```c
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          46 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   2
  On-line CPU(s) list:    0,1
Vendor ID:                GenuineIntel
  Model name:             Intel(R) Xeon(R) CPU @ 2.20GHz
    CPU family:           6
    Model:                79
    Thread(s) per core:   2
    Core(s) per socket:   1
    Socket(s):            1
    Stepping:             0
    BogoMIPS:             4399.99
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 cl
                          flush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc re
                          p_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3
                           fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand
                           hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp 
                          fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm rdseed adx sm
                          ap xsaveopt arat md_clear arch_capabilities
Virtualization features:  
  Hypervisor vendor:      KVM
  Virtualization type:    full
Caches (sum of all):      
  L1d:                    32 KiB (1 instance)
  L1i:                    32 KiB (1 instance)
  L2:                     256 KiB (1 instance)
  L3:                     55 MiB (1 instance)
NUMA:                     
  NUMA node(s):           1
  NUMA node0 CPU(s):      0,1
Vulnerabilities:          
  Gather data sampling:   Not affected
  Itlb multihit:          Not affected
  L1tf:                   Mitigation; PTE Inversion
  Mds:                    Vulnerable; SMT Host state unknown
  Meltdown:               Vulnerable
  Mmio stale data:        Vulnerable
  Reg file data sampling: Not affected
  Retbleed:               Vulnerable
  Spec rstack overflow:   Not affected
  Spec store bypass:      Vulnerable
  Spectre v1:             Vulnerable: __user pointer sanitization and usercopy barriers only; no swa
                          pgs barriers
  Spectre v2:             Vulnerable; IBPB: disabled; STIBP: disabled; PBRSB-eIBRS: Not affected; BH
                          I: Vulnerable (Syscall hardening enabled)
  Srbds:                  Not affected
  Tsx async abort:        Vulnerable
```

We have found those characteristics using this command:

```
!lscpu
```

The VM of google colab is equipped with this GPU:

```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:00:04.0 Off |                    0 |
| N/A   36C    P8               9W /  70W |      0MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

We have found those characteristics using the command:

```bash
!nvidia-smi
```

# Hotspot identification

The nested loops in the `step_kernel_mod` function:

```C
void step_kernel_mod(int ni, int nj, float fact, float* temp_in, float* temp_out) {
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;

  // loop over all points in domain (except boundary)
  for (int j = 1; j < nj - 1; j++) {
    for (int i = 1; i < ni - 1; i++) {
      // find indices into linear memory
      // for central point and neighbours
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i - 1, j);
      ip10 = I2D(ni, i + 1, j);
      i0m1 = I2D(ni, i, j - 1);
      i0p1 = I2D(ni, i, j + 1);

      // evaluate derivatives
      d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
      d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];

      // update temperatures
      temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
    }
  }
}
```

This section iterates over nearly every grid point (excluding the boundaries) on a large `SIZExSIZE` matrix for each of the 200 time steps. Each iteration involves multiple memory accesses, that means reading the central cell and its four neighbours. This segment accesses adjacent elements and performs several arithmetic operations to compute the finite difference update, making the routine both compute-bound and memory-bound.

# Vectorization

```bash
Begin optimization report for: step_kernel_mod

# MAIN HOTSPOT
LOOP BEGIN at hw2/heat_vanilla.c (18, 3)
<Multiversioned v2>
    remark 15319: Loop was not vectorized: novector directive used

    LOOP BEGIN at hw2/heat_vanilla.c (19, 5)
        remark 15319: Loop was not vectorized: novector directive used
    LOOP END
LOOP END
# END MAIN HOTSPOT

LOOP BEGIN at hw2/heat_vanilla.c (18, 3)
<Multiversioned v1>
    remark 15541: loop was not vectorized: outer loop is not an auto-vectorization candidate.

    LOOP BEGIN at hw2/heat_vanilla.c (19, 5)
        remark 15300: LOOP WAS VECTORIZED
        remark 15305: vectorization support: vector length 4
        remark 15389: vectorization support: unmasked unaligned unit stride load: [ /home/leonardo/Github/high-performance-clowns/hw2/heat_vanilla.c (29, 36) ] 
        remark 15389: vectorization support: unmasked unaligned unit stride load: [ /home/leonardo/Github/high-performance-clowns/hw2/heat_vanilla.c (29, 51) ] 
        remark 15389: vectorization support: unmasked unaligned unit stride load: [ /home/leonardo/Github/high-performance-clowns/hw2/heat_vanilla.c (29, 16) ] 
        remark 15389: vectorization support: unmasked unaligned unit stride load: [ /home/leonardo/Github/high-performance-clowns/hw2/heat_vanilla.c (30, 16) ] 
        remark 15389: vectorization support: unmasked unaligned unit stride load: [ /home/leonardo/Github/high-performance-clowns/hw2/heat_vanilla.c (30, 51) ] 
        remark 15389: vectorization support: unmasked unaligned unit stride store: [ /home/leonardo/Github/high-performance-clowns/hw2/heat_vanilla.c (33, 7) ] 
        remark 15475: --- begin vector loop cost summary ---
        remark 15476: scalar cost: 27.000000 
        remark 15477: vector cost: 12.156250 
        remark 15478: estimated potential speedup: 2.187500 
        remark 15309: vectorization support: normalized vectorization overhead 0.234375
        remark 15488: --- end vector loop cost summary ---
        remark 15447: --- begin vector loop memory reference summary ---
        remark 15450: unmasked unaligned unit stride loads: 5 
        remark 15451: unmasked unaligned unit stride stores: 1 
        remark 15474: --- end vector loop memory reference summary ---
    LOOP END

    LOOP BEGIN at hw2/heat_vanilla.c (19, 5)
    <Remainder loop for vectorization>
    LOOP END
LOOP END
```

The vectorization issues in `step_kernel_mod` mainly come from two things:

- **Index Calculation with the I2D Macro**:
    The use of the `I2D` macro makes it harder for the compiler to understand the memory access pattern. This obscurity can prevent the compiler from effectively creating SIMD (vectorized) instructions.

- **Potential Aliasing**:
    The pointers `temp_in` and `temp_out` are not marked with `restrict`. Without `restrict`, the compiler must assume that these pointers might overlap, so it will not optimize the loops as aggressively for vectorization

# Sequential measurements

For retrieving the sequential measurements we have used this code:

```python
SIZES = [1000, 2000, 4000, 6000, 8000, 10000]
times = []

for size in SIZES:
  !make cpu SIZE={size}
  process = subprocess.Popen(['./release/heat'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  stdout, stderr = process.communicate()

  # Extract the time using regex
  match = re.search(r"Elapsed time:\s*([\d.]+)\s*ms", stdout)
  if match:
    time = float(match.group(1))
    times.append(time)
  else:
    print(f"Could not find the time in the output for size {size}")
    print("stdout:", stdout)
    print("stderr:", stderr)
    times.append(None) # append None to keep the list aligned if something goes wrong
```

What we have obtained is this:

| SIZE  | CPU Execution Time (ms) |
|-------|-------------------------|
| 1000  | 229.154                 |
| 2000  | 1330.138                |
| 4000  | 6035.192                |
| 6000  | 13611.214               |
| 8000  | 23880.111               |
| 10000 | 36636.103               |


As we can see, the time needed for the program to process a matrix of size=```10000x10000``` is really high.


# CUDA implementation

To convert the given algorithm into a CUDA kernel function, we replaced the loops inside the functions with direct calculations of `x` and `y` using `blockIdx`, `blockDim`, and `threadIdx`. \
This transformation enables parallel execution, as demonstrated here:

```C
__global__ void step_kernel_mod_dev(const size_t ni, const size_t nj,
                                    const float fact,
                                    const float* temp_in,
                                    float* temp_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i > 0 && i < ni - 1) && (j > 0 && j < nj - 1)) {
    // indices
    size_t ij = I2D(ni, i, j);
    size_t im1j = I2D(ni, i - 1, j);
    size_t ip1j = I2D(ni, i + 1, j);
    size_t ijm1 = I2D(ni, i, j - 1);
    size_t ijp1 = I2D(ni, i, j + 1);

    // second derivatives
    float dx2 = temp_in[ip1j] - 2.0f * temp_in[ij] + temp_in[im1j];
    float dy2 = temp_in[ijp1] - 2.0f * temp_in[ij] + temp_in[ijm1];

    // update
    temp_out[ij] = temp_in[ij] + fact * (dx2 + dy2);
  }
}
```

When launching the kernel, two parameters must be supplied: the number of blocks and the number of threads per block. \
We set the threads per block by defining the block dimensions, and then we compute the number of blocks:

```C
dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
dim3 numBlocks((ni + threadsPerBlock.x - 1) / threadsPerBlock.x, 
               (nj + threadsPerBlock.y - 1) / threadsPerBlock.y);
```

Then the kernel is called by invoking:

```C
step_kernel_mod_dev<<<numBlocks, threadsPerBlock>>>(ni, nj, tfac, temp1_d, temp2_d);
```

# GPU measurements

## Generating data

To generate all the data and plot the results we used this code on the Colab notebook:

```python
def run_cuda_cycle(block_x, block_y, size):
    if not os.path.exists("./src/heat.cu"):
        print("Error: Required CUDA source file is missing.")
        return None

    # Compile CUDA code
    compile_cmd = f"nvcc -O3 -arch=sm_75 -D BLOCKDIM_X={block_x} -D BLOCKDIM_Y={block_y} -D SIZE={size} ./src/heat.cu -o ./src/heat_cuda -run"
    result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)

    # Extract execution time usin,g regex
    match = re.search(r"GPU time:\s+([\d.]+)\s+ms", result.stderr, re.DOTALL)
    if not match:
        match = re.search(r"GPU time:\s+([\d.]+)\s+ms", result.stdout, re.DOTALL)

    if match:
        execution_time = float(match.group(1))
        print(f"Execution Time: {execution_time} ms")
        return execution_time
    else:
        print("Failed to extract execution time.")
        return None
   
# Store results for visualization
results = []

block_x_values = [1, 2, 4, 8, 16, 32]
block_y_values = [1, 2, 4, 8, 16, 32]
size_values = [1000, 10000]

for block_x in block_x_values:
    for block_y in block_y_values:
        for size in size_values:
            print(f"\nRunning with BLOCKDIM_X={block_x}, BLOCKDIM_Y={block_y}, SIZE={size}")
            time_ms = run_cuda_cycle(block_x, block_y, size)
            if time_ms is not None:
                results.append((block_x, block_y, size, time_ms))



# Convert results into NumPy arrays for easy manipulation
results_array = np.array(results)

print(results)
```

## Results

| BLOCKDIM_X | BLOCKDIM_Y | SIZE  | Execution Time (ms) |
|------------|------------|-------|---------------------|
| 1          | 1          | 1000  | 397.41              |
| 1          | 1          | 10000 | 22597.21            |
| 1          | 2          | 1000  | 246.01              |
| 1          | 2          | 10000 | 11428.86            |
| 1          | 4          | 1000  | 152.11              |
| 1          | 4          | 10000 | 6221.48             |
| 1          | 8          | 1000  | 67.13               |
| 1          | 8          | 10000 | 4190.27             |
| 1          | 16         | 1000  | 66.79               |
| 1          | 16         | 10000 | 3964.75             |
| 1          | 32         | 1000  | 70.04               |
| 1          | 32         | 10000 | 4240.1              |
| 2          | 1          | 1000  | 290.89              |
| 2          | 1          | 10000 | 11665.04            |
| 2          | 2          | 1000  | 149.05              |
| 2          | 2          | 10000 | 6243.06             |
| 2          | 4          | 1000  | 78.15               |
| 2          | 4          | 10000 | 3543.88             |
| 2          | 8          | 1000  | 34.64               |
| 2          | 8          | 10000 | 2470.83             |
| 2          | 16         | 1000  | 35.81               |
| 2          | 16         | 10000 | 2483.13             |
| 2          | 32         | 1000  | 35.02               |
| 2          | 32         | 10000 | 2505.36             |
| 4          | 1          | 1000  | 153.51              |
| 4          | 1          | 10000 | 6405.87             |
| 4          | 2          | 1000  | 78.01               |
| 4          | 2          | 10000 | 3541.15             |
| 4          | 4          | 1000  | 39.02               |
| 4          | 4          | 10000 | 2127.01             |
| 4          | 8          | 1000  | 18.3                |
| 4          | 8          | 10000 | 1481.12             |
| 4          | 16         | 1000  | 19.51               |
| 4          | 16         | 10000 | 1680.5              |
| 4          | 32         | 1000  | 20.46               |
| 4          | 32         | 10000 | 1726.68             |
| 8          | 1          | 1000  | 76.98               |
| 8          | 1          | 10000 | 3674.55             |
| 8          | 2          | 1000  | 38.94               |
| 8          | 2          | 10000 | 2042.79             |
| 8          | 4          | 1000  | 19.78               |
| 8          | 4          | 10000 | 1178.2              |
| 8          | 8          | 1000  | 10.26               |
| 8          | 8          | 10000 | 941.68              |
| 8          | 16         | 1000  | 10.36               |
| 8          | 16         | 10000 | 936.52              |
| 8          | 32         | 1000  | 11.18               |
| 8          | 32         | 10000 | 968.99              |
| 16         | 1          | 1000  | 39.4                |
| 16         | 1          | 10000 | 2107.46             |
| 16         | 2          | 1000  | 20.17               |
| 16         | 2          | 10000 | 1255.31             |
| 16         | 4          | 1000  | 9.63                |
| 16         | 4          | 10000 | 784.81              |
| 16         | 8          | 1000  | 9.81                |
| 16         | 8          | 10000 | 782.39              |
| 16         | 16         | 1000  | 10.57               |
| 16         | 16         | 10000 | 804.65              |
| 16         | 32         | 1000  | 12.49               |
| 16         | 32         | 10000 | 875.15              |
| 32         | 1          | 1000  | 20.82               |
| 32         | 1          | 10000 | 1261.63             |
| 32         | 2          | 1000  | 9.54                |
| 32         | 2          | 10000 | 757.64              |
| 32         | 4          | 1000  | 9.28                |
| 32         | 4          | 10000 | 752.1               |
| 32         | 8          | 1000  | 9.72                |
| 32         | 8          | 10000 | 741.55              |
| 32         | 16         | 1000  | 11.2                |
| 32         | 16         | 10000 | 831.59              |
| 32         | 32         | 1000  | 14.04               |
| 32         | 32         | 10000 | 907.58              |

These data represent the execution times (in milliseconds) for cuda tested under various configurations. 
Each row details:
- BLOCKDIM_X and BLOCKDIM_Y: Parameters likely representing the dimensions of the computational block (commonly used in parallel processing or GPU computing).
- SIZE: The problem size, with values of 1000 and 10000.
- Execution Time: The measured time to complete the execution for that configuration.

The results indicate that increasing the problem size generally leads to longer execution times. Additionally, variations in the block dimensions significantly affect performance, suggesting that tuning these parameters can optimize computational efficiency.

<div style="display: flex; justify-content: center; align-items: center; width: 100%;">
  <figure style="display: flex; flex-direction: row; justify-content: center; align-items: center;">
    <img src="./images/results.png" alt="OMP" width="100%" />
  </figure>
</div>


## CUDA transfers calls

The profiling results show that a significant amount of time is consumed by API calls, particularly those related to synchronization and memory transfers. For instance, the call to cudaEventSynchronize accounts for about 53.90% of the API call time, indicating that waiting for the GPU operations to complete is a major factor in overall performance. Similarly, the cudaMemcpy operations (both HtoD and DtoH) also contribute substantially, taking up around 31.63% of the API call time. Although the kernel execution (step_kernel_mod_dev) represents 63.16% of the GPU activity, these overheads from synchronization and memory transfers are noteworthy and can be a target for further performance optimization.


This is the result of using ```nvprof``` with 10000 size:
```c
==45897== Profiling application: ./src/heat_cuda
==45897== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   63.16%  899.87ms       200  4.4993ms  4.0265ms  6.8937ms  step_kernel_mod_dev(unsigned long, unsigned long, float, float const *, float*)
                   24.13%  343.73ms         1  343.73ms  343.73ms  343.73ms  [CUDA memcpy DtoH]
                   12.71%  181.14ms         2  90.572ms  89.753ms  91.391ms  [CUDA memcpy HtoD]
      API calls:   53.90%  899.17ms         1  899.17ms  899.17ms  899.17ms  cudaEventSynchronize
                   31.63%  527.64ms         3  175.88ms  90.004ms  345.02ms  cudaMemcpy
                   14.15%  236.09ms         2  118.04ms  254.29us  235.83ms  cudaMalloc
                    0.17%  2.8580ms         2  1.4290ms  432.90us  2.4251ms  cudaFree
                    0.07%  1.0862ms       200  5.4300us  3.3440us  249.90us  cudaLaunchKernel
                    0.06%  1.0660ms         2  532.98us  4.7750us  1.0612ms  cudaEventRecord
                    0.01%  184.23us       114  1.6160us     125ns  65.752us  cuDeviceGetAttribute
                    0.00%  30.569us         2  15.284us  1.0410us  29.528us  cudaEventCreate
                    0.00%  21.590us         1  21.590us  21.590us  21.590us  cuDeviceGetName
                    0.00%  11.857us         1  11.857us  11.857us  11.857us  cuDeviceGetPCIBusId
                    0.00%  7.2590us         1  7.2590us  7.2590us  7.2590us  cudaEventElapsedTime
                    0.00%  5.5550us         2  2.7770us     719ns  4.8360us  cudaEventDestroy
                    0.00%  3.2840us         3  1.0940us     160ns  2.8660us  cuDeviceGetCount
                    0.00%  1.2670us         2     633ns     146ns  1.1210us  cuDeviceGet
                    0.00%     630ns         1     630ns     630ns     630ns  cuModuleGetLoadingMode
                    0.00%     498ns         1     498ns     498ns     498ns  cuDeviceTotalMem
                    0.00%     383ns         1     383ns     383ns     383ns  cuDeviceGetUuid

```


This is result of using ```nvprof``` with 1000 size, as we can see the api call impact a lot more than before:

```c
==46887== Profiling application: ./src/heat_cuda
==46887== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   79.24%  14.724ms       200  73.618us  71.039us  74.750us  step_kernel_mod_dev(unsigned long, unsigned long, float, float const *, float*)
                   12.36%  2.2959ms         1  2.2959ms  2.2959ms  2.2959ms  [CUDA memcpy DtoH]
                    8.41%  1.5619ms         2  780.93us  763.80us  798.07us  [CUDA memcpy HtoD]
      API calls:   91.78%  246.91ms         2  123.45ms  116.47us  246.79ms  cudaMalloc
                    5.01%  13.475ms         1  13.475ms  13.475ms  13.475ms  cudaEventSynchronize
                    2.28%  6.1409ms         3  2.0470ms  1.0237ms  4.0451ms  cudaMemcpy
                    0.61%  1.6483ms       200  8.2410us  5.5560us  242.75us  cudaLaunchKernel
                    0.19%  517.76us         2  258.88us  255.07us  262.69us  cudaFree
                    0.09%  239.42us       114  2.1000us     206ns  87.270us  cuDeviceGetAttribute
                    0.01%  26.674us         2  13.337us  1.9590us  24.715us  cudaEventCreate
                    0.01%  23.079us         1  23.079us  23.079us  23.079us  cuDeviceGetName
                    0.01%  18.301us         2  9.1500us  6.8520us  11.449us  cudaEventRecord
                    0.00%  10.752us         1  10.752us  10.752us  10.752us  cuDeviceGetPCIBusId
                    0.00%  7.2120us         1  7.2120us  7.2120us  7.2120us  cudaEventElapsedTime
                    0.00%  4.8870us         2  2.4430us     896ns  3.9910us  cudaEventDestroy
                    0.00%  3.7620us         3  1.2540us     250ns  2.9930us  cuDeviceGetCount
                    0.00%  1.3370us         2     668ns     257ns  1.0800us  cuDeviceGet
                    0.00%     859ns         1     859ns     859ns     859ns  cuModuleGetLoadingMode
                    0.00%     591ns         1     591ns     591ns     591ns  cuDeviceTotalMem
                    0.00%     412ns         1     412ns     412ns     412ns  cuDeviceGetUuid

```

# Conclusions
