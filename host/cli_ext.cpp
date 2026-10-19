#include "cli_ext.h"
#include <cstdio>
int cli_ext(const std::string& cmd, int, char**) { std::fprintf(stderr, "unknown command %s\n", cmd.c_str()); return 2; }
