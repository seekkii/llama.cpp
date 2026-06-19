#pragma once

#include "common.h"
#include "server-context.h"
#include "server-task.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

struct llama_cli_stream_chunk {
    std::string text;
    std::string reasoning_text;

    bool empty() const {
        return text.empty() && reasoning_text.empty();
    }
};

struct llama_cli_completion_params {
    int32_t max_tokens = 256;
    float temperature = 0.80f;
    float top_p = 0.95f;
    int32_t top_k = 40;
    float min_p = 0.05f;
    float repeat_penalty = 1.00f;
};

struct llama_cli_response_state {
    std::atomic<bool> done = false;
    std::atomic<bool> cancel_requested = false;
    std::atomic<bool> cancelled = false;
    std::mutex mutex;
    std::condition_variable condition;
    std::string content;
    std::string reasoning_content;
    std::vector<std::string> content_chunks;
    std::vector<std::string> reasoning_chunks;
    std::vector<llama_cli_stream_chunk> stream_chunks;
    std::string error;
    result_timings timings;
};

class llama_cli_response {
public:
    llama_cli_response() = default;
    explicit llama_cli_response(std::shared_ptr<llama_cli_response_state> state);

    bool done() const;
    bool cancelled() const;
    bool has_error() const;
    bool wait(int timeout_ms = -1) const;
    std::string text() const;
    std::string reasoning_text() const;
    std::vector<std::string> chunks() const;
    std::vector<std::string> reasoning_chunks() const;
    std::string error() const;
    std::string result() const;

private:
    friend class llama_cli_stream;
    std::shared_ptr<llama_cli_response_state> state;
};

class llama_cli_stream {
public:
    llama_cli_stream() = default;
    llama_cli_stream(llama_cli_response response, std::string model_name);

    bool done() const;
    bool wait(int timeout_ms = -1) const;
    llama_cli_stream_chunk next_chunk(int timeout_ms = -1);
    std::string next_text(int timeout_ms = -1);
    std::string result() const;
    const std::string & model_name() const;

private:
    llama_cli_response response;
    std::string model_name_value;
    size_t next_chunk_index = 0;
};

class llama_cli_engine {
public:
    explicit llama_cli_engine(const std::vector<std::string> & args);
    ~llama_cli_engine();

    llama_cli_response send(const std::string & prompt, const llama_cli_completion_params & params = {});
    std::string complete(const std::string & prompt, const llama_cli_completion_params & params = {});
    llama_cli_stream stream(const std::string & prompt, const llama_cli_completion_params & params = {});
    std::string add_image(const std::string & path);
    std::string add_image_bytes(raw_buffer data);

    void clear();
    void close();

    bool is_busy() const;
    std::string model_name() const;
    std::string history_json() const;

private:
    common_params params;
    server_context ctx_server;
    json messages = json::array();
    std::vector<raw_buffer> input_files;
    task_params defaults;
    std::string system_prompt;
    std::thread inference_thread;
    std::thread worker_thread;
    mutable std::mutex mutex;
    std::shared_ptr<llama_cli_response_state> active_request;
    bool closed = false;

    static void init_runtime();
    static common_params parse_args(const std::vector<std::string> & args);
    static std::vector<char *> make_argv(std::vector<std::string> & args);

    void reset_messages_locked();
    common_chat_params format_chat_locked() const;
    server_task make_completion_task_locked(server_response_reader & reader, const llama_cli_completion_params & completion_params) const;
    void run_request(std::shared_ptr<llama_cli_response_state> state, std::unique_ptr<server_response_reader> reader);
    void join_finished_worker();
};