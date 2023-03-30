#ifndef AscentAdaptor_h
#define AscentAdaptor_h

#include "AscentAdaptor.h"
#include "data.h"
#include <ascent/ascent.hpp>
#include "conduit_blueprint.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

namespace AscentAdaptor
{
  ascent::Ascent ascent;
  conduit::Node mesh;
  conduit::Node actions; // default actions can also be overidden by file ascent_actions.yaml

void Initialize()
{
  using data::options;
  using data::x_new;
  
  std::cout << "AscentInitialize.........................................\n";
  conduit::Node ascent_options;
  ascent.open(ascent_options);

  mesh["coordsets/coords/dims/i"].set(options.nx);
  mesh["coordsets/coords/dims/j"].set(options.ny);
  // do not specify the 3rd dimension with a dim of 1, a z_origin, and a z_spacing

  mesh["coordsets/coords/origin/x"].set(0.0);
  mesh["coordsets/coords/origin/y"].set(0.0);
  mesh["coordsets/coords/type"].set("uniform");

  float spacing = 1.0/(options.nx+1.0);
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
  mesh["fields/temperature/values"].set_external(x_new.device_data(), options.nx * options.ny);
  
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

  //std::cout << mesh.to_yaml();
}

void Execute() //int cycle, double time, Grid& grid, Attributes& attribs)
{
  using data::options;
  mesh["state/cycle"].set(options.timestep);
  mesh["state/time"].set(options.timestep * options.dt);

  ascent.publish(mesh);
  ascent.execute(actions);
}

void Finalize()
{
  ascent.close();
  std::cout << "AscentFinalize.........................................\n";
}
}
#endif
