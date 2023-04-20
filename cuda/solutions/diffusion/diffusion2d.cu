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

#ifdef USE_CATALYST
#include <catalyst.hpp>
#include "catalyst_conduit_blueprint.hpp"
#include <catalyst_conduit.hpp>
using namespace conduit_cpp;
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

int main(int argc, char** argv) {
    // set up parameters
    // first argument is the y dimension = 2^arg
    size_t pow    = read_arg(argc, argv, 1, 8);
    // second argument is the number of time steps
    size_t nsteps = read_arg(argc, argv, 2, 100);

#ifdef USE_CATALYST
// consume the first two original arguments
// and scan the catalyst-specific args
  conduit_cpp::Node node;
  for (int cc = 2; cc < argc; ++cc)
    {
    if (strcmp(argv[cc], "--output") == 0 && (cc + 1) < argc)
      {
      node["catalyst/pipelines/0/type"].set("io");
      node["catalyst/pipelines/0/filename"].set(argv[cc + 1]);
      node["catalyst/pipelines/0/channel"].set("grid");
      ++cc;
      }
    else if (strcmp(argv[cc], "--pv") == 0 && (cc + 1) < argc)
      {
      const auto path = std::string(argv[cc + 1]);
      node["catalyst/scripts/script0/filename"].set_string(path);
      ++cc;
      }
    }

  // indicate that we want to load ParaView-Catalyst
  node["catalyst_load/implementation"].set_string("paraview");
  // search path should be indicated via the env variable CATALYST_IMPLEMENTATION_PATHS
  // node["catalyst_load/search_paths/paraview"] = PARAVIEW_IMPL_DIR;

  catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
  if (err != catalyst_status_ok)
  {
    std::cerr << "ERROR: Failed to initialize Catalyst: " << err << std::endl;
  }
  else
  {
    conduit_cpp::Node about_info;
    catalyst_about(conduit_cpp::c_node(&about_info));
    std::cout << about_info.to_yaml() ;
    std::cout << "CatalystInitialize.........................................\n";
  }
  
  conduit_cpp::Node catalyst;
  auto catalyst_state = catalyst["catalyst/state"];

  // Add channels.
  // We only have 1 channel here. Let's name it 'grid'.
  auto channel = catalyst["catalyst/channels/grid"];

  // Since this example is using Conduit Mesh Blueprint to define the mesh,
  // we set the channel's type to "mesh".
  channel["type"].set("mesh");
  auto mesh = channel["data"];
  
#endif

#ifdef USE_ASCENT
  ascent::Ascent ascent;
  conduit::Node mesh, actions;
  std::cout << "AscentInitialize.........................................\n";
  ascent.open();
#endif

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
    
#if defined USE_ASCENT || defined USE_CATALYST
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
  mesh["fields/temperature/values"].set_external(x0, nx * ny);

#endif

#ifdef USE_CATALYST
 conduit_cpp::Node verify_info;
 if (!conduit_cpp::BlueprintMesh::verify(mesh, verify_info))
   {
   CATALYST_CONDUIT_ERROR("blueprint verify failed!")
   }
#endif

#ifdef USE_ASCENT
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
        diffusion<<<grid_dim, block_dim>>>(x0, x1, nx, ny, dt);
        std::swap(x0, x1);
#ifdef USE_ASCENT
        if(!(step % 50)) {
          mesh["state/cycle"].set(step);
          mesh["state/time"].set(step * dt);
          ascent.publish(mesh);
          ascent.execute(actions);
        }
#endif
#ifdef USE_CATALYST
        catalyst_state["timestep"].set(step);
        catalyst_state["time"].set(step * dt);
        copy_to_host<double>(x1, x_host, buffer_size);
        catalyst_status err = catalyst_execute(conduit_cpp::c_node(&catalyst));
        if (err != catalyst_status_ok)
          std::cerr << "ERROR: Failed to execute Catalyst: " << err << std::endl;
#endif
    }

    auto stop_event = stream.enqueue_event();
    stop_event.wait();

    copy_to_host<double>(x0, x_host, buffer_size);

#ifdef USE_ASCENT
  ascent.close();
  std::cout << "AscentFinalize.........................................\n";
#endif

#ifdef USE_CATALYST
    conduit_cpp::Node nodef;
    err = catalyst_finalize(conduit_cpp::c_node(&nodef));
    if (err != catalyst_status_ok)
      std::cerr << "ERROR: Failed to finalize Catalyst: " << err << std::endl;
    std::cout << "CatalystFinalize.........................................\n";
#endif

    double time = stop_event.time_since(start_event);

    std::cout << "## " << time << "s, "
              << nsteps*(nx-2)*(ny-2) / time << " points/second"
              << std::endl << std::endl;

    std::cout << "writing to output.bin/bov" << std::endl;
    write_to_file(nx, ny, x_host);
    
    free(x_host);
    cudaFree(x0);
    cudaFree(x1);

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
