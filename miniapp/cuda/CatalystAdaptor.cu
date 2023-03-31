#ifndef CatalystAdaptor_h
#define CatalystAdaptor_h

#include "CatalystAdaptor.h"
#include "data.h"
#include <catalyst.hpp>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

namespace CatalystAdaptor
{
void Initialize(int argc, char* argv[])
{
  conduit_cpp::Node node;
  for (int cc = 0; cc < argc; ++cc)
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
    else{
    //std::cerr << "skipping the original arg: " << argv[cc] << std::endl;
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
}

void Execute()
{
  using data::options;
  using data::x_new;
  conduit_cpp::Node exec_params;

  // add time/cycle information
  auto state = exec_params["catalyst/state"];
  state["timestep"].set(options.timestep);
  state["time"].set(options.timestep * options.dt);

  // Add channels.
  // We only have 1 channel here. Let's name it 'grid'.
  auto channel = exec_params["catalyst/channels/grid"];

  // Since this example is using Conduit Mesh Blueprint to define the mesh,
  // we set the channel's type to "mesh".
  channel["type"].set("mesh");

  // now create the mesh.
  auto mesh = channel["data"];

  //std::cout << "Uniform Grid dimensions =[" << (sim.local_extents[1] - sim.local_extents[0] + 1) << ", " << (sim.local_extents[3] - sim.local_extents[2] + 1) << ", 1]"<< std::endl;
  mesh["coordsets/coords/dims/i"].set(options.nx);
  mesh["coordsets/coords/dims/j"].set(options.ny);
  mesh["coordsets/coords/dims/k"].set(1);
    
  //std::cout << "Uniform Grid Origin =[" << sim.cx[0] << ", " << sim.cy[0] << ", 0.]"<< std::endl;
  mesh["coordsets/coords/origin/x"].set(0.0);
  mesh["coordsets/coords/origin/y"].set(0.0);
  mesh["coordsets/coords/origin/z"].set(0.0);
  mesh["coordsets/coords/type"].set("uniform");

  float spacing = 1.0/(options.nx+1.0);
  mesh["coordsets/coords/spacing/dx"].set(spacing);
  mesh["coordsets/coords/spacing/dy"].set(spacing);
  mesh["coordsets/coords/spacing/dz"].set(spacing);

  // add topology.
  mesh["topologies/mesh/type"].set("uniform");
  mesh["topologies/mesh/coordset"].set("coords");

  // Finally, add fields.
  auto fields = mesh["fields"];
  // temperature is vertex-data.
  fields["temperature/association"].set("vertex");
  fields["temperature/type"].set("scalar");
  fields["temperature/topology"].set("mesh");
  fields["temperature/volume_dependent"].set("false");
  // Conduit supports zero copy, allowing a Conduit Node to describe and
  // point to externally allocated data
  fields["temperature/values"].set_external(x_new.host_data(), options.nx * options.ny);

  catalyst_status err = catalyst_execute(conduit_cpp::c_node(&exec_params));
  if (err != catalyst_status_ok)
  {
    std::cerr << "ERROR: Failed to execute Catalyst: " << err << std::endl;
  }
}

void Finalize()
{
  conduit_cpp::Node node;
  catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
  if (err != catalyst_status_ok)
  {
    std::cerr << "ERROR: Failed to finalize Catalyst: " << err << std::endl;
  }
}
}

#endif
