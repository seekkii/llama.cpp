#pragma once

#include <string>
#include <vector>

int llama_cli_run(int argc, char ** argv);
int llama_cli_run_script(int argc, char ** argv, const std::vector<std::string> & scripted_inputs);
int llama_cli_run_args(const std::vector<std::string> & args);
int llama_cli_run_script_args(const std::vector<std::string> & args, const std::vector<std::string> & scripted_inputs);