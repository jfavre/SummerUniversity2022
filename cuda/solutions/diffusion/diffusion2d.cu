#include <iostream>
#include <fstream>
#include <cstdio>

#include <cuda.h>

#include "util.hpp"
#include "cuda_stream.hpp"

#ifdef USE_ASCENT
#include <ascent/ascent.hpp>
#include "conduit_blueprint.hpp"
#endif
// 2D diffusion example
// the grid has a fixed width of nx=128
// the use specifies the height, ny, as a power of two
// note that nx and ny have 2 added to them to account for halos

template <typename T>
void fill_gpu(T *v, T value, int n);

void write_to_file(int nx, int ny, double* data);

__global__
void diffusion(double *x0, double *x1, int nx, int ny, double dt) {
    int i = threadIdx.x + blockDim.x*blockIdx.x + 1;
    int j = threadIdx.y + blockDim.y*blockIdx.y + 1;

    if (i<nx-1 && j<ny-1) {
        int pos = nx*j + i;
          x1[pos] = x0[pos] + dt * (-4.*x0[pos]
                     + x0[pos-1] + x0[pos+1]
                     + x0[pos-nx] + x0[pos+nx]);

    }
}
// TODO : implement stencil using 2d launch configuration
// NOTE : i-major ordering, i.e. x[i,j] is indexed at location [i+j*nx]
//  for(i=1; i<nx-1; ++i) {
//    for(j=1; j<ny-1; ++j) {
//        x1[i,j] = x0[i,j] + dt * (-4.*x0[i,j]
//                   + x0[i,j-1] + x0[i,j+1]
//                   + x0[i-1,j] + x0[i+1,j]);
//    }
//  }

int main(int argc, char** argv) {
    // set up parameters
    // first argument is the y dimension = 2^arg
    size_t pow    = read_arg(argc, argv, 1, 8);
    // second argument is the number of time steps
    size_t nsteps = read_arg(argc, argv, 2, 100);

    // set domain size
    size_t nx = 128+2;
    size_t ny = (1 << pow)+2;
    double dt = 0.1;

    std::cout << "\n## " << nx << "x" << ny
              << " for " << nsteps << " time steps"
              << " (" << nx*ny << " grid points)"
              << std::endl;

    // allocate memory on device and host
    // note : allocate enough memory for the halo around the boundary
    auto buffer_size = nx*ny;
    double *x_host = malloc_host<double>(buffer_size);
    double *x0     = malloc_device<double>(buffer_size);
    double *x1     = malloc_device<double>(buffer_size);
#ifdef USE_ASCENT
  ascent::Ascent ascent;
  conduit::Node mesh, actions;
  std::cout << "AscentInitialize.........................................\n";
  ascent.open();

  mesh["coordsets/coords/dims/i"].set(nx);
  mesh["coordsets/coords/dims/j"].set(ny);
  // do not specify the 3rd dimension with a dim of 1, a z_origin, and a z_spacing

  mesh["coordsets/coords/origin/x"].set(0.0);
  mesh["coordsets/coords/origin/y"].set(0.0);
  mesh["coordsets/coords/type"].set("uniform");

  float spacing = 1.0/(nx+1.0);
  mesh["coordsets/coords/spacing/dx"].set(spacing);
  mesh["coordsets/coords/spacing/dy"].set(spacing);
  
  // add topology.
  mesh["topologies/mesh/type"].set("uniform");
  mesh["topologies/mesh/coordset"].set("coords");

  // temperature is vertex-data.
  mesh["fields/temperature/association"].set("vertex");
  mesh["fields/temperature/type"].set("scalar");
  mesh["fields/temperature/topology"].set("mesh");
  mesh["fields/temperature/volume_dependent"].set("false");
  mesh["fields/temperature/values"].set_external(x1, nx * ny);
  
  conduit::Node verify_info;
  if (!conduit::blueprint::mesh::verify(mesh, verify_info))
    {
    // verify failed, print error message
    CONDUIT_INFO("blueprint verify failed!" + verify_info.to_json());
    }
  else CONDUIT_INFO("blueprint verify success!");

  conduit::Node &add_action = actions.append();
  
  add_action["action"] = "add_scenes";
  conduit::Node &scenes       = add_action["scenes"];
  scenes["view/plots/p1/type"]  = "pseudocolor";
  scenes["view/plots/p1/field"] = "temperature";
  scenes["view/image_prefix"] = "temperature_%04d";
#endif
    // set initial conditions of 0 everywhere
    fill_gpu(x0, 0., buffer_size);
    fill_gpu(x1, 0., buffer_size);

    // set boundary conditions of 1 on south border
    fill_gpu(x0, 1., nx);
    fill_gpu(x1, 1., nx);
    fill_gpu(x0+nx*(ny-1), 1., nx);
    fill_gpu(x1+nx*(ny-1), 1., nx);

    cuda_stream stream;
    cuda_stream copy_stream();
    auto start_event = stream.enqueue_event();

    // grid and block config
    auto find_num_blocks = [](int x, int bdim) {return (x+bdim-1)/bdim;};
    dim3 block_dim(16, 16);
    int nbx = find_num_blocks(nx-2, block_dim.x);
    int nby = find_num_blocks(ny-2, block_dim.y);
    dim3 grid_dim(nbx, nby);

    // time stepping loop
    for(auto step=0; step<nsteps; ++step) {
        // TODO: launch the diffusion kernel in 2D
        diffusion<<<grid_dim, block_dim>>>(x0, x1, nx, ny, dt);
        std::swap(x0, x1);
#ifdef USE_ASCENT
        if(!(step % 50)) {
          mesh["state/cycle"].set(step);
          mesh["state/time"].set(dt);
          ascent.publish(mesh);
          ascent.execute(actions);
        }
#endif
    }

    auto stop_event = stream.enqueue_event();
    stop_event.wait();

    copy_to_host<double>(x0, x_host, buffer_size);

#ifdef USE_ASCENT
  ascent.close();
  std::cout << "AscentFinalize.........................................\n";
#endif

    double time = stop_event.time_since(start_event);

    std::cout << "## " << time << "s, "
              << nsteps*(nx-2)*(ny-2) / time << " points/second"
              << std::endl << std::endl;

    std::cout << "writing to output.bin/bov" << std::endl;
    write_to_file(nx, ny, x_host);

    return 0;
}

template <typename T>
__global__
void fill(T *v, T value, int n) {
    int tid  = threadIdx.x + blockDim.x*blockIdx.x;

    if(tid<n) {
        v[tid] = value;
    }
}

template <typename T>
void fill_gpu(T *v, T value, int n) {
    auto block_dim = 192ul;
    auto grid_dim = n/block_dim + (n%block_dim ? 1 : 0);

    fill<T><<<grid_dim, block_dim>>>(v, value, n);
}

void write_to_file(int nx, int ny, double* data) {
    {
        FILE* output = fopen("output.bin", "w");
        fwrite(data, sizeof(double), nx * ny, output);
        fclose(output);
    }

    std::ofstream fid("output.bov");
    fid << "TIME: 0.0" << std::endl;
    fid << "DATA_FILE: output.bin" << std::endl;
    fid << "DATA_SIZE: " << nx << " " << ny << " 1" << std::endl;;
    fid << "DATA_FORMAT: DOUBLE" << std::endl;
    fid << "VARIABLE: phi" << std::endl;
    fid << "DATA_ENDIAN: LITTLE" << std::endl;
    fid << "CENTERING: nodal" << std::endl;
    fid << "BRICK_SIZE: 1.0 1.0 1.0" << std::endl;
}
