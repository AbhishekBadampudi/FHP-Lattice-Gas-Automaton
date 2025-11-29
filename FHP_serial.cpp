/* 
============================ Serial C++ Implementation =============================
Simulating a Lattice Gas Automaton, which doesn't solve the traditional Differential Equations(like Navier- Stokes) directly, rather simulate the particles hopping on a grid.
Hexa grid preserves the mathematical symmetry of the Navier-Stokes equations for the fluid flow. 
The 4 collision rules below preserve Mass(no. of particles) and Momentum (sum of velocity vectors).
 - Rule 1: Two particles crash head-on. To conserve momentum (net 0), they must bounce out head-on in a different direction.
 - Rule 2: Three particles meet at 120 degrees. They bounce back, effectively rotating the setup.
 - Rule 3: Four particles meet. This is mathematically the "inverse" of Rule 1 (imagine holes instead of particles).
 - Rule 4: Two particles crash head-on, but a 3rd "spectator" particle is watching. The spectator flies through, while the crashing pair rotates.
 */

/* 
============================ Serial C++ Implementation =============================
Simulating a Lattice Gas Automaton
*/

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

// ============================================================================
// CONSTANTS & CONFIGURATION
// ============================================================================

const int NX = 300;
const int NY = 100;
const int NUM_DIRS = 6;

// Hexagonal Lattice Directions (0=E, 1=NE, 2=NW, 3=W, 4=SW, 5=SE)
const double DIR_X[6] = {1.0, 0.5, -0.5, -1.0, -0.5, 0.5};
const double DIR_Y[6] = {0.0, 0.86602540378, 0.86602540378, 0.0, -0.86602540378, -0.86602540378};

// Boundary Types
enum BoundaryType { FLUID = 0, NO_SLIP = 1, SLIP = 2 };

// Struct to hold node state
struct Node {
    int bits[6]; // 0 or 1 for each direction
    int type;    // BoundaryType

    Node() : type(FLUID) {
        for(int i=0; i<6; ++i) bits[i] = 0;
    }
};

// Global Grid
std::vector<Node> grid;
std::vector<Node> next_grid;

// Simulation Params
double obstacle_x, obstacle_y, obstacle_r;
std::mt19937 rng;
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

inline int idx(int x, int y) {
    return x * NY + y;
}

void get_neighbor(int x, int y, int d, int& nx, int& ny) {
    int parity = y % 2;
    int dx[] = {1, parity, parity-1, -1, parity-1, parity};
    int dy[] = {0, 1, 1, 0, -1, -1};
    nx = x + dx[d];
    ny = y + dy[d];
}

// ============================================================================
// COLLISION
// ============================================================================

void collision() {
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            int i = idx(x, y);

            // Preserve type information
            next_grid[i].type = grid[i].type;

            // Skip solid nodes
            if (grid[i].type == NO_SLIP) {
                for(int d = 0; d < 6; ++d) next_grid[i].bits[d] = 0;
                continue;
            }

            // Count particles
            int count = 0;
            int occ[6] = {0};
            for (int d = 0; d < 6; ++d) {
                occ[d] = grid[i].bits[d];
                count += occ[d];
            }

            // Default: no change
            for(int d = 0; d < 6; ++d) next_grid[i].bits[d] = grid[i].bits[d];

            // Apply collision rules
            double rand_val = uniform_dist(rng);

            if (count == 2) {
                // Rule 1: Head-on collision
                for (int d = 0; d < 3; ++d) {
                    if (occ[d] && occ[d+3]) {
                        if (rand_val < 0.5) {
                            next_grid[i].bits[d] = 0; next_grid[i].bits[d+3] = 0;
                            next_grid[i].bits[(d+1)%6] = 1; next_grid[i].bits[(d+4)%6] = 1;
                        } else {
                            next_grid[i].bits[d] = 0; next_grid[i].bits[d+3] = 0;
                            next_grid[i].bits[(d+5)%6] = 1; next_grid[i].bits[(d+2)%6] = 1;
                        }
                        break;
                    }
                }
            }
            else if (count == 3) {
                // Rule 2: Symmetric 3-Particle
                int s1 = occ[0] + occ[2] + occ[4];
                int s2 = occ[1] + occ[3] + occ[5];

                if (s1 == 3) {
                    next_grid[i].bits[0] = 0; next_grid[i].bits[2] = 0; next_grid[i].bits[4] = 0;
                    next_grid[i].bits[1] = 1; next_grid[i].bits[3] = 1; next_grid[i].bits[5] = 1;
                } else if (s2 == 3) {
                    next_grid[i].bits[1] = 0; next_grid[i].bits[3] = 0; next_grid[i].bits[5] = 0;
                    next_grid[i].bits[0] = 1; next_grid[i].bits[2] = 1; next_grid[i].bits[4] = 1;
                } else {
                    // Rule 4: Head-on + spectator
                    for (int d = 0; d < 3; ++d) {
                        if (occ[d] && occ[d+3]) {
                        
                            // Found the head-on pair (d, d+3)
            
            		    // We need to check which of the other axes holds the spectator.
            		    // The other axes are (d+1)/(d+4) and (d+5)/(d+2).
            		    
            		    // check if axis 1(d+1 or d+4) is occupied by the spectator
            		    bool axis1_blocked = occ[(d+1)%6] || occ[(d+4)%6];
            		    
            		    next_grid[i].bits[d] = 0;
            		    next_grid[i].bits[d+3] = 0;
            		    
                            if (axis1_blocked) {
                                // Spectator is on axis 1, so we must rotate to axis 2
                                next_grid[i].bits[(d+5)%6] = 1; next_grid[i].bits[(d+2)%6] = 1;
                                //next_grid[i].bits[(d+1)%6] = 1; next_grid[i].bits[(d+4)%6] = 1;
                            } else {
                            // Spectator must be on axis 2; so we must rotate to axis 1
                                next_grid[i].bits[(d+1)%6] = 1; next_grid[i].bits[(d+4)%6] = 1;
                                //next_grid[i].bits[(d+5)%6] = 1; next_grid[i].bits[(d+2)%6] = 1;
                            }
                            break;
                        }
                    }
                }
            }
            else if (count == 4) {
                // Rule 3: Head-on of holes
                int holes[6];
                for(int d=0; d < 6; ++d) holes[d] = !occ[d];

                for (int d = 0; d < 3; ++d) {
                    if (holes[d] && holes[d+3]) {
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

// ============================================================================
// STREAMING
// ============================================================================

void streaming() {
    // Clear destination bits (preserve types)
    for(int i = 0; i < NX * NY; ++i) {
        for(int d = 0; d < 6; ++d) grid[i].bits[d] = 0;
    }

    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            int src_idx = idx(x, y);

            if (next_grid[src_idx].type == NO_SLIP) continue;

            for (int d = 0; d < 6; ++d) {
                if (next_grid[src_idx].bits[d] == 1) {
                    int nx, ny;
                    get_neighbor(x, y, d, nx, ny);

                    // Periodic X
                    if (nx < 0) nx = NX - 1;
                    if (nx >= NX) nx = 0;

                    // Slip Walls Y
                    if (ny < 0 || ny >= NY) {
                        int reflect_map[] = {0, 5, 4, 3, 2, 1};
                        int ref_d = reflect_map[d];
                        grid[src_idx].bits[ref_d] = 1;
                        continue;
                    }

                    // Obstacles (no-slip)
                    int target_idx = idx(nx, ny);
                    if (next_grid[target_idx].type == NO_SLIP) {
                        int bounce_d = (d + 3) % 6;
                        grid[src_idx].bits[bounce_d] = 1;
                    } else {
                        grid[target_idx].bits[d] = 1;
                    }
                }
            }
        }
    }
}

// ============================================================================
// SETUP & OUTPUT
// ============================================================================

void init_simulation(double density) {
    grid.resize(NX * NY);
    next_grid.resize(NX * NY);

    std::cout << "Initializing Grid: " << NX << "x" << NY << "\n";

    // Setup obstacle and boundaries
    obstacle_x = NX / 2.0;
    obstacle_y = NY / 2.0;
    obstacle_r = NY / 5.0;

    for(int x = 0; x < NX; ++x) {
        for(int y= 0; y < NY; ++y) {
            int i = idx(x, y);

            if (y == 0 || y == NY - 1) grid[i].type = SLIP;
            else grid[i].type = FLUID;

            double dx = x - obstacle_x;
            double dy = y - obstacle_y;
            if (std::sqrt(dx*dx + dy*dy) <= obstacle_r) {
                grid[i].type = NO_SLIP;
            }
        }
    }

    // Initialize with STRONG rightward flow
    for(int i = 0; i < NX * NY; ++i) {
        if (grid[i].type != FLUID) continue;

        for(int d = 0; d < 6; ++d) {
            double threshold = density;
            // STRONG bias for rightward directions
            if (d == 0) threshold += 0.25;      // E (right)
            if (d == 1 || d == 5) threshold += 0.15;  // NE, SE
            
            if(uniform_dist(rng) < threshold) {
                grid[i].bits[d] = 1;
            }
        }
    }
}

void save_csv(int step) {
    std::ostringstream filename_stream;
    filename_stream << "output/fhp_" << std::setfill('0') << std::setw(5) << step << ".csv";
    std::string filename = filename_stream.str();
    
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // HEADER
    outfile << "x,y,u_avg,v_avg,density,is_solid\n";
    
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            int i = idx(x, y);
            
            // Calculate basic stats even for obstacles (just zeros) to keep grid shape
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
            
            // For raw data, save sum or average. Average is fine if density > 0
            double u_avg = (count > 0) ? (vx / count) : 0.0;
            double v_avg = (count > 0) ? (vy / count) : 0.0;
            
            outfile << x << "," << y << "," << u_avg << "," << v_avg << "," << count << "," << is_solid << "\n";
        }
    }
    outfile.close();
}


int main() {
    // Seed RNG
    rng.seed(12345);  // Same seed for reproducibility
    
    if (!fs::exists("output")) {
        fs::create_directory("output");
    }

    init_simulation(0.25);  // Set density to 25%

    const int MAX_STEPS = 2000;
    const int SAVE_INTERVAL = 50;

    std::cout << "Starting Simulation (" << MAX_STEPS << " steps)...\n";

    for (int t = 0; t <= MAX_STEPS; ++t) {
        collision();
        streaming();
        
        // Copy grid back to next_grid for next iteration
        // This preserves particles in the system!
        /* for (int i = 0; i < NX * NY; ++i) {
            next_grid[i] = grid[i];
        }

*/
        if (t % SAVE_INTERVAL == 0) {
            save_csv(t);
            
            // Count total particles for debugging
            int total_particles = 0;
            for (int i = 0; i < NX * NY; ++i) {
                if (grid[i].type == FLUID) {
                    for (int d = 0; d < 6; ++d) {
                        total_particles += grid[i].bits[d];
                    }
                }
            }
            std::cout << "Step " << t << " saved. Total particles: " << total_particles << "\n";
        }
    }

    std::cout << "Done! Results in 'output/' folder.\n";
    return 0;
}
