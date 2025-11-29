/* 
============================ CUDA C++ Implementation =============================
FHP Lattice Gas Automaton - GPU Accelerated
Each thread processes one grid node.

Compilation commands:

nvcc -O3 -std=c++17 FHP_cuda.cu -o fhp_cuda
./fhp_cuda

*/

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <filesystem>
#include <curand_kernel.h>

namespace fs = std::filesystem;

// =======================================================================
// CONSTANTS
// =======================================================================
const int NX = 300;
const int NY = 100;
const int NUM_DIRS = 6;

// Hexagonal Lattice Directions (0=E, 1=NE, 2=NW, 3=W, 4=SW, 5=SE)
__constant__ float DIR_X[6] = {1.0f, 0.5f, -0.5f, -1.0f, -0.5f, 0.5f};
__constant__ float DIR_Y[6] = {0.0f, 0.86602540378f, 0.86602540378f, 0.0f, -0.86602540378f, -0.86602540378f};

// Boundary Types
enum BoundaryType { FLUID = 0, NO_SLIP = 1, SLIP = 2 };

// Node struct (GPU-friendly: no constructor, plain data)
struct Node {
    int bits[6];
    int type;
};

// ========================================================================
// DEVICE HELPER FUNCTIONS
// ========================================================================

__device__ inline int idx(int x, int y) {
    return x * NY + y;
}

__device__ void get_neighbor(int x, int y, int d, int& nx, int& ny) {
    int parity = y % 2;
    int dx[] = {1, parity, parity-1, -1, parity-1, parity};
    int dy[] = {0, 1, 1, 0, -1, -1};
    nx = x + dx[d];
    ny = y + dy[d];
}

// ======================================================================
// CUDA KERNELS
// ======================================================================

// Initialize RNG states (one per thread)
__global__ void init_rng_kernel(curandState* states, unsigned long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= NX || y >= NY) return;
    
    int idx = x * NY + y;
    curand_init(seed, idx, 0, &states[idx]);
}

// Initialize grid with flow bias
__global__ void init_grid_kernel(Node* grid, curandState* states, float density, float obstacle_x, float obstacle_y, float obstacle_r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= NX || y >= NY) return;
    
    int i = idx(x, y);
    curandState localState = states[i];
    
    // Set boundary types
    if (y == 0 || y == NY - 1) grid[i].type = SLIP;
    else grid[i].type = FLUID;
    
    // Obstacle (circle)
    float dx = x - obstacle_x;
    float dy = y - obstacle_y;
    if (sqrtf(dx*dx + dy*dy) <= obstacle_r) {
        grid[i].type = NO_SLIP;
    }
    
    // Initialize flow with rightward bias
    for (int d = 0; d < 6; ++d) {
        grid[i].bits[d] = 0;
    }
    
    if (grid[i].type == FLUID) {
        for (int d = 0; d < 6; ++d) {
            float threshold = density;
            if (d == 0) threshold += 0.25f;  // East
            if (d == 1 || d == 5) threshold += 0.15f;  // NE, SE
            
            if (curand_uniform(&localState) < threshold) {
                grid[i].bits[d] = 1;
            }
        }
    }
    
    states[i] = localState;
}

// Collision kernel
__global__ void collision_kernel(Node* grid, Node* next_grid, curandState* states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= NX || y >= NY) return;
    
    int i = idx(x, y);
    curandState localState = states[i];
    
    next_grid[i].type = grid[i].type;
    
    // Skip solid nodes
    if (grid[i].type == NO_SLIP) {
        for (int d = 0; d < 6; ++d) next_grid[i].bits[d] = 0;
        return;
    }
    
    // Count particles
    int occ[6];
    int count = 0;
    for (int d = 0; d < 6; ++d) {
        occ[d] = grid[i].bits[d];
        count += occ[d];
        next_grid[i].bits[d] = occ[d];  // Default (no change)
    }
    
    float rand_val = curand_uniform(&localState);
    
    // Rule 1: 2-particle head-on
    if (count == 2) {
        for (int d = 0; d < 3; ++d) {
            if (occ[d] && occ[d+3]) {
                next_grid[i].bits[d] = 0;
                next_grid[i].bits[d+3] = 0;
                if (rand_val < 0.5f) {
                    next_grid[i].bits[(d+1)%6] = 1;
                    next_grid[i].bits[(d+4)%6] = 1;
                } else {
                    next_grid[i].bits[(d+5)%6] = 1;
                    next_grid[i].bits[(d+2)%6] = 1;
                }
                break;
            }
        }
    }
    // Rule 2 & 4: 3-particle
    else if (count == 3) {
        int s1 = occ[0] + occ[2] + occ[4];
        int s2 = occ[1] + occ[3] + occ[5];
        
        if (s1 == 3) {
            next_grid[i].bits[0]=0; next_grid[i].bits[2]=0; next_grid[i].bits[4]=0;
            next_grid[i].bits[1]=1; next_grid[i].bits[3]=1; next_grid[i].bits[5]=1;
        } else if (s2 == 3) {
            next_grid[i].bits[1]=0; next_grid[i].bits[3]=0; next_grid[i].bits[5]=0;
            next_grid[i].bits[0]=1; next_grid[i].bits[2]=1; next_grid[i].bits[4]=1;
        } else {
            // Rule 4: Spectator
            for (int d = 0; d < 3; ++d) {
                if (occ[d] && occ[d+3]) {
                    bool axis1_blocked = occ[(d+1)%6] || occ[(d+4)%6];
                    next_grid[i].bits[d] = 0;
                    next_grid[i].bits[d+3] = 0;
                    if (axis1_blocked) {
                        next_grid[i].bits[(d+5)%6] = 1;
                        next_grid[i].bits[(d+2)%6] = 1;
                    } else {
                        next_grid[i].bits[(d+1)%6] = 1;
                        next_grid[i].bits[(d+4)%6] = 1;
                    }
                    break;
                }
            }
        }
    }
    // Rule 3: 4-particle
    else if (count == 4) {
        int holes[6];
        for (int d = 0; d < 6; ++d) holes[d] = !occ[d];
        
        for (int d = 0; d < 3; ++d) {
            if (holes[d] && holes[d+3]) {
                next_grid[i].bits[(d+1)%6] = 0;
                next_grid[i].bits[(d+4)%6] = 0;
                next_grid[i].bits[(d+5)%6] = 0;
                next_grid[i].bits[(d+2)%6] = 0;
                
                if (rand_val < 0.5f) {
                    next_grid[i].bits[d] = 1;
                    next_grid[i].bits[d+3] = 1;
                } else {
                    next_grid[i].bits[d] = 1;
                    next_grid[i].bits[d+3] = 1;
                }
                break;
            }
        }
    }
    
    states[i] = localState;
}

// Streaming kernel
__global__ void streaming_kernel(Node* grid, Node* next_grid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= NX || y >= NY) return;
    
    int src_idx = idx(x, y);
    
    // Clear destination
    for (int d = 0; d < 6; ++d) grid[src_idx].bits[d] = 0;
    
    __syncthreads();  // Ensure all threads cleared before writing
    
    if (next_grid[src_idx].type == NO_SLIP) return;
    
    // Stream particles
    for (int d = 0; d < 6; ++d) {
        if (next_grid[src_idx].bits[d] == 1) {
            int nx, ny;
            get_neighbor(x, y, d, nx, ny);
            
            // Periodic X
            if (nx < 0) nx = NX - 1;
            if (nx >= NX) nx = 0;
            
            // Slip Y boundaries
            if (ny < 0 || ny >= NY) {
                int reflect_map[] = {0, 5, 4, 3, 2, 1};
                atomicAdd(&grid[src_idx].bits[reflect_map[d]], 1);
                continue;
            }
            
            int target_idx = idx(nx, ny);
            
            // No-slip obstacle
            if (next_grid[target_idx].type == NO_SLIP) {
                atomicAdd(&grid[src_idx].bits[(d+3)%6], 1);
            } else {
                atomicAdd(&grid[target_idx].bits[d], 1);
            }
        }
    }
}

// ==========================================================================
// HOST CODE
// ==========================================================================

void save_csv(const std::vector<Node>& grid, int step) {
    std::ostringstream filename;
    filename << "output/fhp_cuda_" << std::setfill('0') << std::setw(5) << step << ".csv";
    
    std::ofstream outfile(filename.str());
    outfile << "x,y,u_avg,v_avg,density,is_solid\n";
    
    float dir_x_host[6] = {1.0f, 0.5f, -0.5f, -1.0f, -0.5f, 0.5f};
    float dir_y_host[6] = {0.0f, 0.86602540378f, 0.86602540378f, 0.0f, -0.86602540378f, -0.86602540378f};
    
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            int i = x * NY + y;
            
            int is_solid = (grid[i].type == NO_SLIP) ? 1 : 0;
            float vx = 0.0f, vy = 0.0f;
            int count = 0;
            
            if (!is_solid) {
                for (int d = 0; d < 6; ++d) {
                    if (grid[i].bits[d]) {
                        vx += dir_x_host[d];
                        vy += dir_y_host[d];
                        count++;
                    }
                }
            }
            
            float u_avg = (count > 0) ? (vx / count) : 0.0f;
            float v_avg = (count > 0) ? (vy / count) : 0.0f;
            
            outfile << x << "," << y << "," << u_avg << "," << v_avg << "," << count << "," << is_solid << "\n";
        }
    }
    outfile.close();
}

int main() {
    // Create output directory
    if (!fs::exists("output")) fs::create_directory("output");
    
    std::cout << "Starting CUDA FHP Simulation (" << NX << "x" << NY << ")\n";
    
    // Allocate device memory
    Node *d_grid, *d_next_grid;
    curandState *d_states;
    
    size_t grid_bytes = NX * NY * sizeof(Node);
    
    cudaMalloc(&d_grid, grid_bytes);
    cudaMalloc(&d_next_grid, grid_bytes);
    cudaMalloc(&d_states, NX * NY * sizeof(curandState));
    
    // Thread block configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, 
                  (NY + blockSize.y - 1) / blockSize.y);
    
    // Initialize RNG
    init_rng_kernel<<<gridSize, blockSize>>>(d_states, 12345);
    cudaDeviceSynchronize();
    
    // Initialize grid
    float obstacle_x = 150.0f;
    float obstacle_y = 50.0f;
    float obstacle_r = 20.0f;
    init_grid_kernel<<<gridSize, blockSize>>>(d_grid, d_states, 0.25f, obstacle_x, obstacle_y, obstacle_r);
    cudaDeviceSynchronize();
    
    // Host buffer for saving
    std::vector<Node> h_grid(NX * NY);
    
    // Main simulation loop
    const int MAX_STEPS = 2000;
    const int SAVE_INTERVAL = 50;
    
    for (int t = 0; t <= MAX_STEPS; ++t) {
        collision_kernel<<<gridSize, blockSize>>>(d_grid, d_next_grid, d_states);
        streaming_kernel<<<gridSize, blockSize>>>(d_grid, d_next_grid);
        cudaDeviceSynchronize();
        
        if (t % SAVE_INTERVAL == 0) {
            cudaMemcpy(h_grid.data(), d_grid, grid_bytes, cudaMemcpyDeviceToHost);
            save_csv(h_grid, t);
            std::cout << "Step " << t << " saved.\n";
        }
    }
    
    // Cleanup
    cudaFree(d_grid);
    cudaFree(d_next_grid);
    cudaFree(d_states);
    
    std::cout << "Done! Results in 'output/' folder.\n";
    return 0;
}
