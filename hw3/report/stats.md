<!-- ### **sequential (no-vec)**

- INTERVALS: 1_000_000_000
    - Time (s): 6.16
    - GFLOPS: 0.97
    - GINTOPS: 0.32
    - hotspots:
        - loop in main at pi_homework.c:30
            - Time (s): 5.89

- INTERVALS: 10_000_000_000
    - Time (s): 59.16
    - GFLOPS: 1.01
    - GINTOPS: 0.34
    - hotspots:
        - loop in main at pi_homework.c:30
            - Time (s): 58.88 -->

### **sequential (best)**

- INTERVALS: 1_000_000_000
    - Time (s): 0.62
    - GFLOPS: 9.75
    - GINTOPS: 2.11
    - hotspots:
        - loop in main at pi_homework.c:26
            - Time (s): 0.61

- INTERVALS: 10_000_000_000
    - Time (s): 5.87
    - GFLOPS: 10.22
    - GINTOPS: 1.22
    - hotspots:
        - loop in main at pi_homework.c:26
            - Time (s): 5.86

- INTERVALS: 100_000_000_000
    - Time (s): 58.68
    - GFLOPS: 10.22
    - GINTOPS: 1.22
    - hotspots:
        - loop in main at pi_homework.c:26
            - Time (s): 58.67

### **parallel**

- INTERVALS: 1_000_000_000
    - Time (s): 0.11
    - GFLOPS: 53.68
    - GINTOPS: 15.66
    - hotspots:
        - loop in main at pi_homework.c:26
            - Time: 1.070s / 20 cores = 0.05

- INTERVALS: 10_000_000_000
    - Time (s): 0.88
    - GFLOPS: 73.02
    - GINTOPS: 21.30
    - hotspots:
        - loop in main at pi_homework.c:26
            - Time (s): 12.1 / 20 = 0.6

- INTERVALS: 100_000_000_000
    - Time (s): 7.81
    - GFLOPS: 76.85
    - GINTOPS: 22.41
    - hotspots:
        - loop in main at pi_homework.c:26
            - Time (s): 125.683 / 20 = 6.28