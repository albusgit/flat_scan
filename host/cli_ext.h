// Pipeline-stage commands of the host harness (cli_ext.cpp).
#pragma once
#include <string>
#include "npz.h"
#include "resample.h"
int cli_ext(const std::string& cmd, int argc, char** argv);
fs::GridMap load_grid(const std::string& path);
void save_grid(const std::string& path, const fs::GridMap& g, fs::Npz extra);
