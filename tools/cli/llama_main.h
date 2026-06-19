// llama_main.h
#pragma once
#include <vector>
#include <string>
#include "ThreadSafeQueue.h"

using StringOrBytes = std::variant<std::string, std::vector<uint8_t>>;

ThreadSafeQueue<StringOrBytes> output_queue;
int my_main(ThreadSafeQueue<std::string>*in_q,ThreadSafeQueue<std::string>*out_q,int argc, char** argv);
int wrapped_main(ThreadSafeQueue<std::string>*in_q,ThreadSafeQueue<std::string>*out_q, std::vector<std::string>& args);
