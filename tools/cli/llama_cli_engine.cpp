#include "llama_cli_engine.h"

#include "arg.h"
#include "chat.h"
#include "fit.h"
#include "server-common.h"

#include <chrono>
#include <fstream>
#include <stdexcept>

namespace {

json make_user_content_json(const std::string & prompt) {
    const std::string marker = get_media_marker();
    if (prompt.find(marker) == std::string::npos) {
        return prompt;
    }

    json content = json::array();
    size_t position = 0;

    while (position <= prompt.size()) {
        size_t marker_pos = prompt.find(marker, position);
        if (marker_pos == std::string::npos) {
            if (position < prompt.size()) {
                content.push_back({
                    {"type", "text"},
                    {"text", prompt.substr(position)}
                });
            }
            break;
        }

        if (marker_pos > position) {
            content.push_back({
                {"type", "text"},
                {"text", prompt.substr(position, marker_pos - position)}
            });
        }

        content.push_back({
            {"type", "media_marker"},
            {"text", marker}
        });
        position = marker_pos + marker.size();
    }

    return content;
}

std::once_flag g_llama_cli_runtime_flag;

std::string result_error_message(const server_task_result_ptr & result) {
    json err_data = result->to_json();
    if (err_data.contains("message")) {
        return err_data["message"].get<std::string>();
    }
    return err_data.dump();
}

void append_response_chunk(
    const std::shared_ptr<llama_cli_response_state> & state,
    const std::string & content_delta,
    const std::string & reasoning_delta) {

    std::lock_guard<std::mutex> lock(state->mutex);

    if (!content_delta.empty()) {
        state->content += content_delta;
        state->content_chunks.push_back(content_delta);
    }

    if (!reasoning_delta.empty()) {
        state->reasoning_content += reasoning_delta;
        state->reasoning_chunks.push_back(reasoning_delta);
    }

    if (!content_delta.empty() || !reasoning_delta.empty()) {
        state->stream_chunks.push_back({content_delta, reasoning_delta});
    }

    state->condition.notify_all();
}

} // namespace

llama_cli_response::llama_cli_response(std::shared_ptr<llama_cli_response_state> state)
    : state(std::move(state)) {
}

llama_cli_stream::llama_cli_stream(llama_cli_response response, std::string model_name)
    : response(std::move(response))
    , model_name_value(std::move(model_name)) {
}

bool llama_cli_stream::done() const {
    return response.done();
}

bool llama_cli_stream::wait(int timeout_ms) const {
    return response.wait(timeout_ms);
}

llama_cli_stream_chunk llama_cli_stream::next_chunk(int timeout_ms) {
    auto state = response.state;
    if (!state) {
        throw std::out_of_range("stream exhausted");
    }

    std::unique_lock<std::mutex> lock(state->mutex);

    auto ready = [this, &state]() {
        return next_chunk_index < state->stream_chunks.size() || state->done.load();
    };

    if (!ready()) {
        if (timeout_ms < 0) {
            state->condition.wait(lock, ready);
        } else if (!state->condition.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready)) {
            return {};
        }
    }

    if (next_chunk_index < state->stream_chunks.size()) {
        return state->stream_chunks[next_chunk_index++];
    }

    throw std::out_of_range("stream exhausted");
}

std::string llama_cli_stream::next_text(int timeout_ms) {
    while (true) {
        llama_cli_stream_chunk chunk = next_chunk(timeout_ms);
        if (chunk.empty()) {
            return {};
        }
        if (!chunk.text.empty() || done()) {
            return chunk.text;
        }
    }
}

std::string llama_cli_stream::result() const {
    return response.result();
}

const std::string & llama_cli_stream::model_name() const {
    return model_name_value;
}

bool llama_cli_response::done() const {
    return state && state->done.load();
}

bool llama_cli_response::cancelled() const {
    return state && state->cancelled.load();
}

bool llama_cli_response::has_error() const {
    return !error().empty();
}

bool llama_cli_response::wait(int timeout_ms) const {
    if (!state) {
        return true;
    }

    if (state->done.load()) {
        return true;
    }

    std::unique_lock<std::mutex> lock(state->mutex);
    if (timeout_ms < 0) {
        state->condition.wait(lock, [this] { return state->done.load(); });
        return true;
    }

    return state->condition.wait_for(
        lock,
        std::chrono::milliseconds(timeout_ms),
        [this] { return state->done.load(); });
}

std::string llama_cli_response::text() const {
    if (!state) {
        return {};
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    return state->content;
}

std::string llama_cli_response::reasoning_text() const {
    if (!state) {
        return {};
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    return state->reasoning_content;
}

std::vector<std::string> llama_cli_response::chunks() const {
    if (!state) {
        return {};
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    return state->content_chunks;
}

std::vector<std::string> llama_cli_response::reasoning_chunks() const {
    if (!state) {
        return {};
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    return state->reasoning_chunks;
}

std::string llama_cli_response::error() const {
    if (!state) {
        return {};
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    return state->error;
}

std::string llama_cli_response::result() const {
    wait(-1);

    if (cancelled()) {
        throw std::runtime_error("request cancelled");
    }

    const std::string err = error();
    if (!err.empty()) {
        throw std::runtime_error(err);
    }

    return text();
}

llama_cli_engine::llama_cli_engine(const std::vector<std::string> & args)
    : params(parse_args(args)) {

    init_runtime();
    llama_numa_init(params.numa);

    defaults.sampling    = params.sampling;
    defaults.speculative = params.speculative;
    defaults.n_keep      = params.n_keep;
    defaults.n_predict   = params.n_predict;
    defaults.antiprompt  = params.antiprompt;
    defaults.stream = true;
    defaults.timings_per_token = true;

    if (!ctx_server.load_model(params)) {
        throw std::runtime_error("failed to load model");
    }

    system_prompt = params.system_prompt;
    reset_messages_locked();

    inference_thread = std::thread([this]() {
        ctx_server.start_loop();
    });
}

llama_cli_engine::~llama_cli_engine() {
    close();
}

llama_cli_response llama_cli_engine::send(const std::string & prompt, const llama_cli_completion_params & completion_params) {
    if (prompt.empty()) {
        throw std::invalid_argument("prompt must not be empty");
    }

    join_finished_worker();

    auto state = std::make_shared<llama_cli_response_state>();
    auto reader = std::make_unique<server_response_reader>(ctx_server.get_response_reader());

    {
        std::lock_guard<std::mutex> lock(mutex);
        if (closed) {
            throw std::runtime_error("engine is closed");
        }
        if (active_request) {
            throw std::runtime_error("engine already has an active request");
        }

        messages.push_back({
            {"role", "user"},
            {"content", prompt}
        });

        server_task task = make_completion_task_locked(*reader, completion_params);
        reader->post_task(std::move(task));
        active_request = state;
    }

    worker_thread = std::thread([this, state, reader = std::move(reader)]() mutable {
        run_request(std::move(state), std::move(reader));
    });

    return llama_cli_response(state);
}

std::string llama_cli_engine::complete(const std::string & prompt, const llama_cli_completion_params & completion_params) {
    auto response = send(prompt, completion_params);
    return response.result();
}

llama_cli_stream llama_cli_engine::stream(const std::string & prompt, const llama_cli_completion_params & completion_params) {
    return llama_cli_stream(send(prompt, completion_params), model_name());
}

std::string llama_cli_engine::add_image(const std::string & path) {
    join_finished_worker();

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open image file: " + path);
    }

    raw_buffer buffer;
    buffer.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    return add_image_bytes(std::move(buffer));
}

std::string llama_cli_engine::add_image_bytes(raw_buffer data) {
    join_finished_worker();

    if (data.empty()) {
        throw std::runtime_error("image data must not be empty");
    }

    std::lock_guard<std::mutex> lock(mutex);
    if (closed) {
        throw std::runtime_error("engine is closed");
    }
    if (active_request) {
        throw std::runtime_error("cannot add an image while a request is active");
    }

    input_files.push_back(std::move(data));
    return get_media_marker();
}

void llama_cli_engine::clear() {
    join_finished_worker();

    std::lock_guard<std::mutex> lock(mutex);
    if (active_request) {
        throw std::runtime_error("cannot clear history while a request is active");
    }
    reset_messages_locked();
    input_files.clear();
}

void llama_cli_engine::close() {
    std::thread worker;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (closed) {
            return;
        }
        closed = true;
        if (active_request) {
            active_request->cancel_requested.store(true);
        }
        worker = std::move(worker_thread);
    }

    if (worker.joinable()) {
        worker.join();
    }

    ctx_server.terminate();
    if (inference_thread.joinable()) {
        inference_thread.join();
    }
}

bool llama_cli_engine::is_busy() const {
    std::lock_guard<std::mutex> lock(mutex);
    return active_request != nullptr;
}

std::string llama_cli_engine::model_name() const {
    return ctx_server.get_meta().model_name;
}

std::string llama_cli_engine::history_json() const {
    std::lock_guard<std::mutex> lock(mutex);
    return messages.dump();
}

void llama_cli_engine::init_runtime() {
    std::call_once(g_llama_cli_runtime_flag, []() {
        common_init();
        llama_backend_init();
    });
}

common_params llama_cli_engine::parse_args(const std::vector<std::string> & args) {
    common_params parsed;
    parsed.verbosity = LOG_LEVEL_ERROR;

    std::vector<std::string> argv_storage;
    argv_storage.reserve(args.size() + 1);
    argv_storage.emplace_back("llama-cli");
    argv_storage.insert(argv_storage.end(), args.begin(), args.end());

    std::vector<char *> argv = make_argv(argv_storage);
    if (!common_params_parse(static_cast<int>(argv_storage.size()), argv.data(), parsed, LLAMA_EXAMPLE_CLI)) {
        throw std::invalid_argument("failed to parse llama-cli arguments");
    }

    if (parsed.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
        throw std::invalid_argument("--no-conversation is not supported by llama-cli");
    }

    return parsed;
}

std::vector<char *> llama_cli_engine::make_argv(std::vector<std::string> & args) {
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);

    for (std::string & arg : args) {
        argv.push_back(arg.data());
    }

    argv.push_back(nullptr);
    return argv;
}

void llama_cli_engine::reset_messages_locked() {
    messages = json::array();
    if (!system_prompt.empty()) {
        messages.push_back({
            {"role", "system"},
            {"content", system_prompt}
        });
    }
}

common_chat_params llama_cli_engine::format_chat_locked() const {
    auto meta = ctx_server.get_meta();
    auto & chat_params = meta.chat_params;

    auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());

    common_chat_templates_inputs inputs;
    inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
    inputs.tools                 = {};
    inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_NONE;
    inputs.json_schema           = "";
    inputs.grammar               = "";
    inputs.use_jinja             = chat_params.use_jinja;
    inputs.parallel_tool_calls   = caps["supports_parallel_tool_calls"];
    inputs.add_generation_prompt = true;
    inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
    inputs.force_pure_content    = chat_params.force_pure_content;
    inputs.enable_thinking       = chat_params.enable_thinking ? common_chat_templates_support_enable_thinking(chat_params.tmpls.get()) : false;

    return common_chat_templates_apply(chat_params.tmpls.get(), inputs);
}

server_task llama_cli_engine::make_completion_task_locked(server_response_reader & reader, const llama_cli_completion_params & completion_params) const {
    auto chat_params = format_chat_locked();

    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id         = reader.get_new_id();
    task.index      = 0;
    task.params     = defaults;
    task.params.n_predict = completion_params.max_tokens;
    task.params.sampling.temp = completion_params.temperature;
    task.params.sampling.top_p = completion_params.top_p;
    task.params.sampling.top_k = completion_params.top_k;
    task.params.sampling.min_p = completion_params.min_p;
    task.params.sampling.penalty_repeat = completion_params.repeat_penalty;
    task.cli_prompt = chat_params.prompt;
    task.cli_files  = input_files;
    task.cli        = true;

    task.params.chat_parser_params = common_chat_parser_params(chat_params);
    task.params.chat_parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
    if (!chat_params.parser.empty()) {
        task.params.chat_parser_params.parser.load(chat_params.parser);
    }

    if (!chat_params.thinking_end_tag.empty()) {
        const llama_vocab * vocab = llama_model_get_vocab(
            llama_get_model(ctx_server.get_llama_context()));

        task.params.sampling.reasoning_budget_tokens = defaults.sampling.reasoning_budget_tokens;
        task.params.sampling.generation_prompt = chat_params.generation_prompt;

        if (!chat_params.thinking_start_tag.empty()) {
            task.params.sampling.reasoning_budget_start =
                common_tokenize(vocab, chat_params.thinking_start_tag, false, true);
        }
        task.params.sampling.reasoning_budget_end =
            common_tokenize(vocab, chat_params.thinking_end_tag, false, true);
        task.params.sampling.reasoning_budget_forced =
            common_tokenize(vocab, defaults.sampling.reasoning_budget_message + chat_params.thinking_end_tag, false, true);
    }

    return task;
}

void llama_cli_engine::run_request(std::shared_ptr<llama_cli_response_state> state, std::unique_ptr<server_response_reader> reader) {
    std::string final_content;
    bool push_assistant_message = false;

    while (true) {
        server_task_result_ptr result = reader->next([state]() {
            return state->cancel_requested.load();
        });

        if (!result) {
            state->cancelled.store(true);
            break;
        }

        if (result->is_error()) {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->error = result_error_message(result);
            break;
        }

        if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
            for (const auto & diff : partial->oaicompat_msg_diffs) {
                append_response_chunk(state, diff.content_delta, diff.reasoning_content_delta);
            }

            std::lock_guard<std::mutex> lock(state->mutex);
            state->timings = partial->timings;
            continue;
        }

        if (auto * final = dynamic_cast<server_task_result_cmpl_final *>(result.get())) {
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->timings = final->timings;
                if (state->content.empty() && !final->content.empty()) {
                    state->content = final->content;
                }
                if (state->reasoning_content.empty() && !final->oaicompat_msg.reasoning_content.empty()) {
                    state->reasoning_content = final->oaicompat_msg.reasoning_content;
                    state->reasoning_chunks.push_back(final->oaicompat_msg.reasoning_content);
                }
                final_content = state->content;
            }
            push_assistant_message = true;
            break;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        if (active_request == state) {
            if (push_assistant_message && !state->cancelled.load() && state->error.empty()) {
                messages.push_back({
                    {"role", "assistant"},
                    {"content", final_content}
                });
            }
            active_request.reset();
        }
    }

    state->done.store(true);
    state->condition.notify_all();
}

void llama_cli_engine::join_finished_worker() {
    std::thread worker;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (worker_thread.joinable() && active_request == nullptr) {
            worker = std::move(worker_thread);
        }
    }

    if (worker.joinable()) {
        worker.join();
    }
}