# rt-micro-benchmarks

Three small, self-contained benchmarks covering the two halves of putting a
learned policy on a robot: getting the model fast enough on the GPU, and
getting the control loop around it to behave deterministically on the CPU.

Each one is independent, runs in minutes, and produces a number and a plot.

| | what it measures | stack |
|---|---|---|
| [01-fused-cuda-kernel](01-fused-cuda-kernel/) | HBM traffic saved by fusing LayerNorm+GELU into one kernel | CUDA, PyTorch extension |
| [02-rt-loop-jitter](02-rt-loop-jitter/) | wake-up jitter of a periodic loop, and what `SCHED_FIFO` / affinity / `mlockall` each buy | C++17, Linux |
| [03-ipc-latency](03-ipc-latency/) | round-trip latency of shared memory vs socketpair vs pipes | C++17, Linux |

## Why these three

A policy that takes 40 ms to run is useless in a 100 Hz loop, so (01) is about
the inference budget. But a policy that runs in 2 ms is *also* useless if the
loop that calls it wakes up 300 µs late one iteration in a thousand, so (02) is
about the loop. And if the safety supervisor lives in a separate process — it
should, so that a hung inference process cannot take the estop with it — then
the cost of talking to it is on the critical path, which is (03).

Together they cover the latency budget of a real control stack end to end.

## Quick start

```bash
# 01 — needs an NVIDIA GPU
cd 01-fused-cuda-kernel
python setup.py build_ext --inplace
python benchmark.py --out results.csv && python plot_results.py

# 02 — Linux. RT modes need privileges.
cd 02-rt-loop-jitter && make
./rt_loop --hz 1000 --sec 10 --out plain.csv
sudo ./rt_loop --hz 1000 --sec 10 --rt 80 --cpu 2 --mlock --out rt.csv
python plot_jitter.py plain.csv rt.csv

# 03 — Linux, needs >= 2 cores for the busy-spin mode
cd 03-ipc-latency && make
make sweep && python plot_latency.py shm.csv socket.csv pipe.csv
```

Requirements: `g++` with C++17, Python with `matplotlib` + `numpy`, and for 01,
a CUDA toolkit and a PyTorch build that matches it.

## Reading the results

Every one of these reports percentiles, not averages. In a control loop the
mean is close to meaningless — what breaks the robot is the one iteration in a
thousand that arrives late. All three tools print p50 / p99 / p99.9 / max and
plot the tail on a log axis for that reason.

## A finding worth keeping

Running 03 inside a 1-core container:

```
mode=shm  (busy-spin)   p50  7999.885 µs
mode=shmy (sched_yield) p50     1.397 µs
mode=socket             p50     3.817 µs
```

Busy-spin shared memory came out ~2000x *slower* than a Unix socket. Nothing
is wrong with the shared-memory path: with one core online, the spinning
parent burns its entire timeslice before the child is ever scheduled to answer.
Spin-waiting is only a latency win when each participant owns a core.

This is the kind of result that quietly poisons a benchmark, so `ipc_latency`
now detects the condition and warns instead of printing a confident wrong
number. Worth remembering before reaching for shared memory on an embedded
target with two cores and a busy scheduler.
