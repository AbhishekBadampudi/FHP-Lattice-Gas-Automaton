/* 
============================ MPI Parallel C++ Implementation =============================
FHP Lattice Gas Automaton with Domain Decomposition
Parallel Strategy: Vertical slices (columns) divided among processes.

execution in a personal pc:

mpic++ -O3 -std=c++17 FHP_MPI.cpp -o fhp_mpi
mpirun -np 4 ./fhp_mpi

*/

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

// ==========================================================================
// CONSTANTS
// ==========================================================================
const int GLOBAL_NX = 300; // Total width
const int NY = 100;        // Height (same for all)
const int NUM_DIRS = 6;

// Hexagonal Lattice Directions (0=E, 1=NE, 2=NW, 3=W, 4=SW, 5=SE)
const double DIR_X[6] = {1.0, 0.5, -0.5, -1.0, -0.5, 0.5};
const double DIR_Y[6] = {0.0, 0.86602540378, 0.86602540378, 0.0, -0.86602540378, -0.86602540378};

// Boundary Types
enum BoundaryType { FLUID = 0, NO_SLIP = 1, SLIP = 2 };

// Struct to hold node state
struct Node {
    int bits[6];
    int type;
    Node() : type(FLUID) { for(int i=0; i<6; ++i) bits[i] = 0; }
};

// =========================================================================
// GLOBAL STATE (Per Process)
// =========================================================================
int rank, num_procs;
int local_nx;      // Width of this process's stripe
int offset_x;      // Global X coordinate where this stripe starts

std::vector<Node> grid;
std::vector<Node> next_grid;

// Ghost Buffers (Left and Right columns)
std::vector<Node> ghost_left(NY);
std::vector<Node> ghost_right(NY);

std::mt19937 rng;
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

double obstacle_x, obstacle_y, obstacle_r;

// =============================================================================
// HELPERS
// =============================================================================

// Local Indexing
inline int idx(int local_x, int y) {
    return local_x * NY + y;
}

// Neighbor Logic (Handles Ghosts)
// Returns TRUE if neighbor is within local domain, FALSE if it needs ghost data
bool get_neighbor_local(int lx, int y, int d, int& nx, int& ny) {
    int parity = y % 2;
    int dx[] = {1, parity, parity-1, -1, parity-1, parity};
    int dy[] = {0, 1, 1, 0, -1, -1};
    
    nx = lx + dx[d];
    ny = y + dy[d];
    
    // Check if inside local domain [0, local_nx-1]
    if (nx >= 0 && nx < local_nx) return true;
    return false;
}

// ==========================================================================
// MPI COMMUNICATION
// ==========================================================================
// Global MPI Datatype for Node
MPI_Datatype MPI_NODE;

void create_mpi_node_type() {
    // We need to serialize/deserialize Node to send via MPI (simplest way)
    struct NodePlain {
        int bits[6];
        int type;
    };
    
    const int count = 2;
    int blocklengths[2] = {6, 1}; 
    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    
    MPI_Aint offsets[2];
    offsets[0] = offsetof(Node, bits);
    offsets[1] = offsetof(Node, type);
    
    MPI_Type_create_struct(count, blocklengths, offsets, types, &MPI_NODE);
    MPI_Type_commit(&MPI_NODE);
}

void exchange_boundaries() {
    int left_rank = (rank - 1 + num_procs) % num_procs;
    int right_rank = (rank + 1) % num_procs;

    // Buffers (Using the actual Node struct, no need for NodePlain)
    std::vector<Node> send_left(NY), send_right(NY);
    
    // We assume ghost_left/right are already allocated to size NY in init.
    // We receive directly into them!

    // Pack Left Column (x=0) to send left
    for(int y=0; y<NY; ++y) {
        send_left[y] = next_grid[idx(0, y)];
    }

    // Pack Right Column (x=local_nx-1) to send right
    for(int y=0; y<NY; ++y) {
        send_right[y] = next_grid[idx(local_nx-1, y)];
    }

    MPI_Request reqs[4];
    
    // Send/Recv using the custom MPI_NODE type
    // Note: count is NY, datatype is MPI_NODE
    
    // 1. Send my Left column to Left Rank
    MPI_Isend(send_left.data(), NY, MPI_NODE, left_rank, 0, MPI_COMM_WORLD, &reqs[0]);
    
    // 2. Send my Right column to Right Rank
    MPI_Isend(send_right.data(), NY, MPI_NODE, right_rank, 1, MPI_COMM_WORLD, &reqs[1]);
    
    // 3. Receive from Right Rank into my Right Ghost
    MPI_Irecv(ghost_right.data(), NY, MPI_NODE, right_rank, 0, MPI_COMM_WORLD, &reqs[2]);
    
    // 4. Receive from Left Rank into my Left Ghost
    MPI_Irecv(ghost_left.data(), NY, MPI_NODE, left_rank, 1, MPI_COMM_WORLD, &reqs[3]);

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
}


// ========================================================================
// PHYSICS
// ========================================================================

void collision() {
    for (int x = 0; x < local_nx; ++x) {
        for (int y = 0; y < NY; ++y) {
            int i = idx(x, y);

            next_grid[i].type = grid[i].type;

            if (grid[i].type == NO_SLIP) {
                for(int d=0; d<6; ++d) next_grid[i].bits[d] = 0;
                continue;
            }

            int occ[6];
            int count = 0;
            for (int d = 0; d < 6; ++d) {
                occ[d] = grid[i].bits[d];
                count += occ[d];
                next_grid[i].bits[d] = occ[d]; // Default
            }

            double rand_val = uniform_dist(rng);

            // Collision Rules
            if (count == 2) {
                for (int d = 0; d < 3; ++d) {
                    if (occ[d] && occ[d+3]) {
                        next_grid[i].bits[d] = 0; next_grid[i].bits[d+3] = 0;
                        if (rand_val < 0.5) {
                            next_grid[i].bits[(d+1)%6] = 1; next_grid[i].bits[(d+4)%6] = 1;
                        } else {
                            next_grid[i].bits[(d+5)%6] = 1; next_grid[i].bits[(d+2)%6] = 1;
                        }
                        break;
                    }
                }
            }
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
                     // Rule 4
                     for (int d = 0; d < 3; ++d) {
                        if (occ[d] && occ[d+3]) {
                            bool axis1_blocked = occ[(d+1)%6] || occ[(d+4)%6];
                            next_grid[i].bits[d] = 0; next_grid[i].bits[d+3] = 0;
                            if (axis1_blocked) {
                                next_grid[i].bits[(d+5)%6] = 1; next_grid[i].bits[(d+2)%6] = 1;
                            } else {
                                next_grid[i].bits[(d+1)%6] = 1; next_grid[i].bits[(d+4)%6] = 1;
                            }
                            break;
                        }
                     }
                }
            }
            else if (count == 4) {
                 int holes[6]; for(int d=0; d<6; ++d) holes[d] = !occ[d];
                 for (int d = 0; d < 3; ++d) {
                    if (holes[d] && holes[d+3]) {
                        next_grid[i].bits[(d+1)%6] = 0; next_grid[i].bits[(d+4)%6] = 0;
                        next_grid[i].bits[(d+5)%6] = 0; next_grid[i].bits[(d+2)%6] = 0;
                        if (rand_val < 0.5) {
                            next_grid[i].bits[d] = 1; next_grid[i].bits[d+3] = 1;
                            next_grid[i].bits[(d+1)%6] = 0; next_grid[i].bits[(d+4)%6] = 0;
                        } else {
                            next_grid[i].bits[d] = 1; next_grid[i].bits[d+3] = 1;
                            next_grid[i].bits[(d+5)%6] = 0; next_grid[i].bits[(d+2)%6] = 0;
                        }
                        break;
                    }
                 }
            }
        }
    }
}

void streaming() {
    // Clear destination
    for(int i=0; i<local_nx*NY; ++i) {
        for(int d=0; d<6; ++d) grid[i].bits[d] = 0;
    }

    // Exchange boundaries (fills ghost_left and ghost_right with data from neighbors)
    exchange_boundaries();

    for (int x = 0; x < local_nx; ++x) {
        for (int y = 0; y < NY; ++y) {
            int src_idx = idx(x, y);
            
            // Check the SOURCE node (next_grid)
            // Wait! In FHP we iterate over TARGET or SOURCE?
            // Your serial code iterates over SOURCE and pushes to TARGET.
            // Let's stick to that.
            
            if (next_grid[src_idx].type == NO_SLIP) continue;

            for (int d = 0; d < 6; ++d) {
                if (next_grid[src_idx].bits[d] == 1) {
                    int nx, ny; // Relative to local_nx
                    int parity = y % 2;
                    int dx[] = {1, parity, parity-1, -1, parity-1, parity};
                    int dy[] = {0, 1, 1, 0, -1, -1};
                    
                    nx = x + dx[d];
                    ny = y + dy[d];
                    
                    // Handle Y Boundaries (Global)
                    if (ny < 0 || ny >= NY) {
                         int reflect_map[] = {0, 5, 4, 3, 2, 1};
                         grid[src_idx].bits[reflect_map[d]] = 1;
                         continue;
                    }

                    // Handle X Boundaries (Ghost / Periodic)
                    // Case 1: Particle stays inside local domain
                    if (nx >= 0 && nx < local_nx) {
                         int target_idx = idx(nx, ny);
                         if (next_grid[target_idx].type == NO_SLIP) {
                             grid[src_idx].bits[(d+3)%6] = 1; // Bounce
                         } else {
                             grid[target_idx].bits[d] = 1; // Move
                         }
                    }
                    // Case 2: Particle goes RIGHT (into neighbor)
                    // We don't write to neighbor's memory. We ignore it.
                    // The neighbor will "pull" it? No, push streaming is hard in MPI.
                    
                    /* 
                       CRITICAL CHANGE FOR MPI: 
                       It is much easier to iterate over TARGET nodes (Pull Scheme)
                       or handle boundary writes carefully.
                       
                       Actually, let's stick to PUSH but ignore writes that go off-screen.
                       BUT: How do we receive particles coming IN?
                       
                       Ah, we need to process the GHOST buffers.
                       The ghost buffers contain the 'next_grid' state of the neighbor's edge.
                    */
                }
            }
        }
    }
    
    // PROCESS GHOSTS (Particles entering from neighbors)
    // Left Ghost (x = -1)
    for(int y=0; y<NY; ++y) {
        if(ghost_left[y].type == NO_SLIP) continue; // Should not happen
        for(int d=0; d<6; ++d) {
            if(ghost_left[y].bits[d]) {
                // Calculate where it lands in our domain
                // It comes from x=-1. 
                int parity = y%2;
                int dx = (d==0 || d==1 || d==5) ? 1 : ((d==3)? -1 : 0); // Rough check
                // Actually reuse neighbor logic
                int dx_val[] = {1, parity, parity-1, -1, parity-1, parity};
                int dy_val[] = {0, 1, 1, 0, -1, -1};
                
                int nx = -1 + dx_val[d]; // Relative to our start
                int ny = y + dy_val[d];
                
                if (nx >= 0 && nx < local_nx && ny >= 0 && ny < NY) {
                     // It entered our domain!
                     int target_idx = idx(nx, ny);
                     if (next_grid[target_idx].type == NO_SLIP) {
                         // Bounces back to ghost... ignore (it's neighbor's problem)
                     } else {
                         grid[target_idx].bits[d] = 1;
                     }
                }
            }
        }
    }
    
    // Right Ghost (x = local_nx)
    for(int y=0; y<NY; ++y) {
         if(ghost_right[y].type == NO_SLIP) continue;
         for(int d=0; d<6; ++d) {
            if(ghost_right[y].bits[d]) {
                int parity = y%2;
                int dx_val[] = {1, parity, parity-1, -1, parity-1, parity};
                int dy_val[] = {0, 1, 1, 0, -1, -1};
                
                int nx = local_nx + dx_val[d]; // Start from right edge
                // Wait, ghost_right is the column AT x=local_nx
                // So relative to us, it is at x=local_nx.
                
                // Correct.
                // If particle moves LEFT (dx < 0), it might enter.
                
                // Actually, ghost_right is the neighbor's x=0.
                // It is effectively at our x=local_nx.
                
                int target_x = local_nx + dx_val[d]; 
                // This logic is getting tricky. 
                
                // Let's simplify:
                // A particle at ghost_right (x=local_nx) moving West (d=3) lands at:
                // x = local_nx - 1. (Inside our domain).
                // We write to grid[idx(local_nx-1, y)].
                
                // Re-calculate target
                int final_x = local_nx + dx_val[d]; // relative to local 0
                int final_y = y + dy_val[d];
                
                if (final_x >= 0 && final_x < local_nx && final_y >=0 && final_y < NY) {
                    int t_idx = idx(final_x, final_y);
                    if(next_grid[t_idx].type != NO_SLIP) {
                        grid[t_idx].bits[d] = 1;
                    }
                }
            }
         }
    }
}

// ============================================================================
// INITIALIZATION & SAVE
// ============================================================================

void init_simulation(double density) {
    // Calculate Decomposition
    local_nx = GLOBAL_NX / num_procs;
    int remainder = GLOBAL_NX % num_procs;
    
    // Simple load balancing
    if (rank < remainder) {
        local_nx++;
        offset_x = rank * local_nx;
    } else {
        offset_x = rank * local_nx + remainder;
    }

    grid.resize(local_nx * NY);
    next_grid.resize(local_nx * NY);

    obstacle_x = GLOBAL_NX / 2.0;
    obstacle_y = NY / 2.0;
    obstacle_r = NY / 5.0;

    for(int x = 0; x < local_nx; ++x) {
        for(int y= 0; y < NY; ++y) {
            int global_x = offset_x + x;
            int i = idx(x, y);

            if (y == 0 || y == NY - 1) grid[i].type = SLIP;
            else grid[i].type = FLUID;

            double dx = global_x - obstacle_x;
            double dy = y - obstacle_y;
            if (std::sqrt(dx*dx + dy*dy) <= obstacle_r) {
                grid[i].type = NO_SLIP;
            }

            // Init Flow
            if (grid[i].type == FLUID) {
                for(int d = 0; d < 6; ++d) {
                    double threshold = density;
                    if (d == 0) threshold += 0.25;
                    if (d == 1 || d == 5) threshold += 0.15;
                    if(uniform_dist(rng) < threshold) grid[i].bits[d] = 1;
                }
            }
        }
    }
}

void save_csv_parallel(int step) {
    std::ostringstream filename_stream;
    // Each rank writes its own file: output/fhp_00100_rank0.csv
    filename_stream << "output/fhp_" << std::setfill('0') << std::setw(5) << step 
                    << "_rank" << rank << ".csv";
    
    std::ofstream outfile(filename_stream.str());
    
    outfile << "x,y,u_avg,v_avg,density,is_solid\n";
    
    for (int x = 0; x < local_nx; ++x) {
        for (int y = 0; y < NY; ++y) {
            int i = idx(x, y);
            int global_x = offset_x + x;
            
            int is_solid = (grid[i].type == NO_SLIP) ? 1 : 0;
            double vx = 0.0, vy = 0.0;
            int count = 0;
            
            if (!is_solid) {
                for (int d = 0; d < 6; ++d) {
                    if (grid[i].bits[d]) {
                        vx += DIR_X[d];
                        vy += DIR_Y[d];
                        count++;
                    }
                }
            }
            
            double u_avg = (count > 0) ? (vx / count) : 0.0;
            double v_avg = (count > 0) ? (vy / count) : 0.0;
            
            outfile << global_x << "," << y << "," << u_avg << "," << v_avg << "," << count << "," << is_solid << "\n";
        }
    }
    outfile.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // 1. Create Custom Type
    create_mpi_node_type();

    if (rank == 0) {
        if (!fs::exists("output")) fs::create_directory("output");
        std::cout << "Starting MPI Simulation on " << num_procs << " processes.\n";
    }
    
    rng.seed(12345 + rank); 
    init_simulation(0.25);

    const int MAX_STEPS = 2000;
    const int SAVE_INTERVAL = 50;

    for (int t = 0; t <= MAX_STEPS; ++t) {
        collision();
        streaming();
        
        if (t % SAVE_INTERVAL == 0) {
            save_csv_parallel(t);
            if (rank == 0) std::cout << "Step " << t << " saved.\n";
        }
    }

    // 2. Clean up Custom Type
    MPI_Type_free(&MPI_NODE);
    
    MPI_Finalize();
    return 0;
}

