#include "cli.h"
#include "llama_cli_engine.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstring>

namespace nb = nanobind;

namespace {

nb::dict make_completion_choice(const std::string & text, nb::handle finish_reason) {
    nb::dict choice;
    choice["text"] = text;
    choice["index"] = 0;
    choice["finish_reason"] = finish_reason;
    return choice;
}

nb::dict make_completion_response(const std::string & model_name, const std::string & text, const std::string & reasoning_text) {
    nb::dict result;
    result["object"] = "text_completion";
    result["model"] = model_name;
    nb::list choices;
    nb::dict choice = make_completion_choice(text, nb::str("stop"));
    choice["reasoning_text"] = nb::str(reasoning_text.c_str());
    choices.append(choice);
    result["choices"] = choices;
    return result;
}

nb::dict make_stream_chunk(const std::string & model_name, const llama_cli_stream_chunk & chunk) {
    nb::dict result;
    result["object"] = "text_completion.chunk";
    result["model"] = model_name;
    nb::list choices;
    nb::dict choice = make_completion_choice(chunk.text, nb::none());
    choice["reasoning_text"] = nb::str(chunk.reasoning_text.c_str());
    choices.append(choice);
    result["choices"] = choices;
    return result;
}

llama_cli_completion_params make_completion_params(
    int32_t max_tokens,
    float temperature,
    float top_p,
    int32_t top_k,
    float min_p,
    float repeat_penalty) {

    llama_cli_completion_params params;
    params.max_tokens = max_tokens;
    params.temperature = temperature;
    params.top_p = top_p;
    params.top_k = top_k;
    params.min_p = min_p;
    params.repeat_penalty = repeat_penalty;
    return params;
}

} // namespace

NB_MODULE(llama_cpp_cli, m) {
    m.doc() = "nanobind Python bindings for llama.cpp's llama-cli";

    nb::class_<llama_cli_response>(m, "LlamaResponse")
        .def("done", &llama_cli_response::done)
        .def("cancelled", &llama_cli_response::cancelled)
        .def("has_error", &llama_cli_response::has_error)
        .def("wait", &llama_cli_response::wait, nb::arg("timeout_ms") = -1, nb::call_guard<nb::gil_scoped_release>())
        .def("text", &llama_cli_response::text)
        .def("reasoning_text", &llama_cli_response::reasoning_text)
        .def("chunks", &llama_cli_response::chunks)
        .def("reasoning_chunks", &llama_cli_response::reasoning_chunks)
        .def("error", &llama_cli_response::error)
        .def("result", &llama_cli_response::result, nb::call_guard<nb::gil_scoped_release>());

    nb::class_<llama_cli_stream>(m, "LlamaStream")
        .def("done", &llama_cli_stream::done)
        .def("wait", &llama_cli_stream::wait, nb::arg("timeout_ms") = -1, nb::call_guard<nb::gil_scoped_release>())
        .def("result", &llama_cli_stream::result, nb::call_guard<nb::gil_scoped_release>())
        .def("next", [](llama_cli_stream & self, int timeout_ms) -> nb::object {
            llama_cli_stream_chunk chunk = self.next_chunk(timeout_ms);
            if (chunk.empty() && !self.done()) {
                return nb::none();
            }
            return nb::cast(make_stream_chunk(self.model_name(), chunk));
        }, nb::arg("timeout_ms") = -1, nb::call_guard<nb::gil_scoped_release>())
        .def("__iter__", [](llama_cli_stream & self) -> llama_cli_stream & {
            return self;
        }, nb::rv_policy::reference_internal)
        .def("__next__", [](llama_cli_stream & self) {
            try {
                return make_stream_chunk(self.model_name(), self.next_chunk(-1));
            } catch (const std::out_of_range &) {
                throw nb::stop_iteration();
            }
        });

    nb::class_<llama_cli_engine>(m, "Llama")
        .def(nb::init<const std::vector<std::string> &>(), nb::arg("args"), nb::call_guard<nb::gil_scoped_release>())
        .def("send", [](llama_cli_engine & self,
                         const std::string & prompt,
                         int32_t max_tokens,
                         float temperature,
                         float top_p,
                         int32_t top_k,
                         float min_p,
                         float repeat_penalty) {
            return self.send(prompt, make_completion_params(max_tokens, temperature, top_p, top_k, min_p, repeat_penalty));
        },
            nb::arg("prompt"),
            nb::kw_only(),
            nb::arg("max_tokens") = 256,
            nb::arg("temperature") = 0.80f,
            nb::arg("top_p") = 0.95f,
            nb::arg("top_k") = 40,
            nb::arg("min_p") = 0.05f,
            nb::arg("repeat_penalty") = 1.00f)
        .def("complete", [](llama_cli_engine & self,
                             const std::string & prompt,
                             int32_t max_tokens,
                             float temperature,
                             float top_p,
                             int32_t top_k,
                             float min_p,
                             float repeat_penalty) {
            return self.complete(prompt, make_completion_params(max_tokens, temperature, top_p, top_k, min_p, repeat_penalty));
        },
            nb::arg("prompt"),
            nb::kw_only(),
            nb::arg("max_tokens") = 256,
            nb::arg("temperature") = 0.80f,
            nb::arg("top_p") = 0.95f,
            nb::arg("top_k") = 40,
            nb::arg("min_p") = 0.05f,
            nb::arg("repeat_penalty") = 1.00f,
            nb::call_guard<nb::gil_scoped_release>())
        .def("stream", [](llama_cli_engine & self,
                           const std::string & prompt,
                           int32_t max_tokens,
                           float temperature,
                           float top_p,
                           int32_t top_k,
                           float min_p,
                           float repeat_penalty) {
            return self.stream(prompt, make_completion_params(max_tokens, temperature, top_p, top_k, min_p, repeat_penalty));
        },
            nb::arg("prompt"),
            nb::kw_only(),
            nb::arg("max_tokens") = 256,
            nb::arg("temperature") = 0.80f,
            nb::arg("top_p") = 0.95f,
            nb::arg("top_k") = 40,
            nb::arg("min_p") = 0.05f,
            nb::arg("repeat_penalty") = 1.00f)
        .def("add_image", &llama_cli_engine::add_image,
            nb::arg("path"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("add_image_bytes", [](llama_cli_engine & self, nb::bytes data) {
            raw_buffer buffer(data.size());
            if (!buffer.empty()) {
                std::memcpy(buffer.data(), data.data(), buffer.size());
            }
            return self.add_image_bytes(std::move(buffer));
        },
            nb::arg("data"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("create_completion", [](llama_cli_engine & self,
                                      const std::string & prompt,
                                      int32_t max_tokens,
                                      float temperature,
                                      float top_p,
                                      int32_t top_k,
                                      float min_p,
                                      float repeat_penalty,
                                      bool stream) -> nb::object {
            auto params = make_completion_params(max_tokens, temperature, top_p, top_k, min_p, repeat_penalty);
            if (stream) {
                return nb::cast(self.stream(prompt, params));
            }
            auto response = self.send(prompt, params);
            const std::string text = response.result();
            return nb::cast(make_completion_response(self.model_name(), text, response.reasoning_text()));
        },
            nb::arg("prompt"),
            nb::kw_only(),
            nb::arg("max_tokens") = 256,
            nb::arg("temperature") = 0.80f,
            nb::arg("top_p") = 0.95f,
            nb::arg("top_k") = 40,
            nb::arg("min_p") = 0.05f,
            nb::arg("repeat_penalty") = 1.00f,
            nb::arg("stream") = false)
        .def("__call__", [](llama_cli_engine & self,
                             const std::string & prompt,
                             int32_t max_tokens,
                             float temperature,
                             float top_p,
                             int32_t top_k,
                             float min_p,
                             float repeat_penalty,
                             bool stream) -> nb::object {
            auto params = make_completion_params(max_tokens, temperature, top_p, top_k, min_p, repeat_penalty);
            if (stream) {
                return nb::cast(self.stream(prompt, params));
            }
            auto response = self.send(prompt, params);
            const std::string text = response.result();
            return nb::cast(make_completion_response(self.model_name(), text, response.reasoning_text()));
        },
            nb::arg("prompt"),
            nb::kw_only(),
            nb::arg("max_tokens") = 256,
            nb::arg("temperature") = 0.80f,
            nb::arg("top_p") = 0.95f,
            nb::arg("top_k") = 40,
            nb::arg("min_p") = 0.05f,
            nb::arg("repeat_penalty") = 1.00f,
            nb::arg("stream") = false)
        .def("clear", &llama_cli_engine::clear)
        .def("close", &llama_cli_engine::close, nb::call_guard<nb::gil_scoped_release>())
        .def("is_busy", &llama_cli_engine::is_busy)
        .def("model_name", &llama_cli_engine::model_name)
        .def("history_json", &llama_cli_engine::history_json);

    m.def(
        "run",
        &llama_cli_run_args,
        nb::arg("args"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Run `llama-cli` with CLI arguments. Expects argv-style arguments without argv[0]."
    );

    m.def(
        "run_script",
        &llama_cli_run_script_args,
        nb::arg("args"),
        nb::arg("inputs"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Run `llama-cli` with CLI arguments and scripted interactive inputs. Each input item is one full user entry or slash command."
    );
}