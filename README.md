# FHP Lattice Gas Automaton: Fluid Flow Simulation

## Project Overview

This project implements the **FHP (Frisch-Hasslacher-Pomeau) Lattice Gas Automaton**, a cellular automaton model for simulating incompressible fluid dynamics. Unlike traditional computational fluid dynamics (CFD) methods that solve differential equations (Navier-Stokes), lattice gas automata simulate fluid flow at the microscopic level by tracking discrete particles hopping on a lattice.

The simulation models **flow around a circular obstacle** in a 2D channel, demonstrating classic fluid phenomena such as:
- Vortex shedding (Kármán vortex street)
- Boundary layer formation
- Wake structures behind bluff bodies

### Implementations

This project provides **three parallel implementations**:

1.  **Serial C++:**  Baseline single-threaded implementation
2.  **MPI (Message Passing Interface):**  Distributed memory parallelization for HPC clusters
3.  **CUDA:**  GPU-accelerated version using NVIDIA GPUs

---

## Physics Background

### The FHP Model

The FHP model overcomes the limitations of earlier lattice gas models (like HPP) by using a **hexagonal lattice** instead of a square grid. This choice is critical because:

> *"The hexagonal grid does not suffer as large anisotropy troubles... a fortunate fact that prompted Frisch to remark that 'the symmetry gods are benevolent.'"* [web:518]

#### Grid Structure
- **300 × 100 hexagonal lattice** (representing a 30 × 10 physical domain with lattice spacing `a = 0.1`)
- Each node has **6 velocity directions**: East, NE, NW, West, SW, SE
- Each direction can be occupied by **0 or 1 particle** (Pauli exclusion principle)

#### Time Evolution

The simulation alternates between two phases every timestep:

**1. Collision Phase**
Particles at the same node interact according to 4 collision rules that conserve **mass** and **momentum**:

- **Rule 1 (2-particle head-on):** Two particles colliding head-on scatter at 60° (random choice between clockwise/counter-clockwise rotation)
- **Rule 2 (3-particle symmetric):** Three particles at 120° intervals reflect to the opposite three directions
- **Rule 3 (4-particle):** Inverse of Rule 1 applied to "holes" (empty directions)
- **Rule 4 (3-particle spectator):** Two particles collide head-on while a third "spectator" passes through; the colliding pair rotates deterministically to avoid the spectator

**2. Streaming Phase**
Each particle moves to the neighboring node in its velocity direction. Boundary conditions are applied:
- **Periodic X-boundaries:** Particles leaving the right edge re-enter from the left (models an infinite cylinder array)
- **Slip Y-boundaries:** Particles reflect specularly off top/bottom walls
- **No-slip obstacle:** Particles bounce back 180° when hitting the circular cylinder

### Connection to Navier-Stokes

Through **Chapman-Enskog expansion**, it can be proven that the macroscopic (averaged) behavior of the FHP model converges to the incompressible Navier-Stokes equations:

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}
$$

where $\nu$ (kinematic viscosity) depends on the collision rules and lattice geometry.

### Lattice Boltzmann Methods

While this project implements a **Lattice Gas Automaton** (discrete boolean particles), the related **Lattice Boltzmann Method (LBM)** uses continuous distribution functions instead. LBM evolved from lattice gas models to:
- Eliminate statistical noise (boolean → real-valued)
- Achieve Galilean invariance
- Improve computational efficiency

However, FHP remains valuable for educational purposes and as a theoretical foundation for understanding emergent hydrodynamics from microscopic rules.

---

## Implementation Details

### Serial Version (`FHP_serial.cpp`)

**Architecture:**
- Single-threaded loop over all grid nodes
- Two grids (`grid`, `next_grid`) for synchronous updates
- Standard C++ with STL containers

**Performance:**
- ~30,000 nodes × 2,000 timesteps = 60 million node updates
- Runtime: ~5-10 minutes on a modern CPU

**Use Case:** Development, debugging, baseline reference

**Compilation & Execution:**
g++ -O3 -std=c++17 FHP_serial.cpp -o fhp_serial
./fhp_serial
---

### MPI Version (`FHP_MPI.cpp`)

**Parallelization Strategy:**
- **Domain Decomposition:** Grid is divided into vertical slices (columns) across MPI ranks
- **Ghost Columns:** Each process exchanges 1 column with left/right neighbors before streaming
- **Custom MPI Datatype:** Uses `MPI_Type_create_struct` to safely transmit `Node` structs across heterogeneous clusters

**Key Features:**
- Cluster-safe (handles compiler padding differences)
- Periodic boundary handling via rank wrap-around: `(rank ± 1) % num_procs`
- Parallel I/O: Each rank writes its own CSV file (`fhp_00100_rank0.csv`, `fhp_00100_rank1.csv`, ...)

**Performance:**
- **Strong Scaling:** Expected near-linear speedup up to ~8-16 cores (limited by communication overhead)
- Ideal for large grids (e.g., 1000×1000) on HPC clusters

**Compilation & Execution:**
mpic++ -O3 -std=c++17 FHP_MPI.cpp -o fhp_mpi
mpirun -np 4 ./fhp_mpi


---

### CUDA Version (`FHP_cuda.cu`)

**Parallelization Strategy:**
- **Massive Parallelism:** Each CUDA thread processes one grid node
- **Thread Grid:** `(16×16)` thread blocks cover the entire `300×100` lattice
- **GPU RNG:** Uses `cuRAND` for on-device random number generation (one state per thread)

**Key Features:**
- **Atomic Operations:** `atomicAdd` in streaming kernel prevents race conditions when multiple particles write to the same target cell
- **Memory Management:** All physics computation on GPU; CPU only used for file I/O every 50 timesteps
- **Coalesced Access:** Memory layout (`x * NY + y`) optimized for GPU memory bandwidth

**Performance:**
- **Expected Speedup:** 10-50× faster than serial (depending on GPU model)
- **Bottleneck:** Atomic operations in streaming (can be mitigated with "pull" scheme)

**Compilation & Execution:**
nvcc -O3 -std=c++17 FHP_cuda.cu -o fhp_cuda
./fhp_cuda

**Requirements:** NVIDIA GPU with Compute Capability ≥ 3.5, CUDA Toolkit 11.0+

---

## File Structure

FHP_Project/
├── FHP_serial.cpp # Serial implementation
├── FHP_MPI.cpp # MPI implementation
├── FHP_cuda.cu # CUDA implementation
├── FHP_visualize.ipynb # Jupyter notebook for visualization
├── README.md # This file
└── output/ # Generated CSV files (created at runtime)
├── fhp_00000.csv # Serial output
├── fhp_00050_rank0.csv # MPI output (multiple files per timestep)
├── fhp_00050_rank1.csv
└── fhp_cuda_00100.csv # CUDA output


---

## Visualization

### Python Notebook (`FHP_visualize.ipynb`)

The visualization pipeline performs **coarse-graining** (spatial averaging) to smooth the noisy particle data into macroscopic flow fields.

**Configuration:**
VERSION = 'serial' # Options: 'serial', 'mpi', 'cuda'
BLOCK_SIZE = 6 # Averaging window (6×6 cells → smooth velocity field)


**Plots Generated:**
1.  **Streamline Plot:** Shows flow direction with color-coded velocity magnitude
2.  **Vorticity Map:** Visualizes rotation (curl of velocity field) to identify vortex structures
3.  **Animation:** Time-evolution video (`.mp4`) of the flow

**Key Functions:**
- `load_and_process_frame()`: Automatically merges MPI rank files or loads single file
- `plot_single_frame()`: Static plot of any timestep
- `create_animation()`: Generates full video

**Dependencies:**

pip install pandas numpy matplotlib


---

## Results & Validation

### Expected Physical Phenomena

For **Reynolds number Re ≈ 100-200** (typical for this setup):
1.  **Steady Wake:** At early times, symmetric wake behind cylinder
2.  **Vortex Shedding:** After ~500 timesteps, alternating vortices detach (Kármán street)
3.  **Periodic Oscillation:** Lift force on cylinder oscillates at Strouhal frequency $f_s \approx 0.2 U/D$

### Verification Methods
- **Mass Conservation:** Total particle count should remain constant (±1% due to initialization randomness)
- **Momentum Conservation:** Check that $\sum \mathbf{v} \approx \text{const}$ in periodic domains
- **Visual Comparison:** Vortex street pattern should match experimental/CFD results

---

## References

1.  **Frisch, U., Hasslacher, B., & Pomeau, Y. (1986).** *Lattice-gas automata for the Navier-Stokes equation.* Physical Review Letters, 56(14), 1505.
2.  **Wolf-Gladrow, D. A. (2005).** *Lattice-Gas Cellular Automata and Lattice Boltzmann Models.* Springer. [web:521]
3.  **Wikipedia: Lattice Gas Automaton** [web:518]
4.  **Wolfram, S. (1986).** *Cellular automaton fluids 1: Basic theory.* Journal of Statistical Physics, 45(3-4), 471-526.

---

## License

This project is developed for educational purposes as part of the Lab Course II (Winter 2023/24) at University.

## Author

Abhishek  
Master's Student in Computer Science (GPU Computing & Scientific Simulation)

---

## Acknowledgments

Special thanks to Dr. T. Korzec, Dr. J. A. Urrea Niño, and Msc. Piyush Kumar for guidance on the FHP model physics and parallel computing strategies.
