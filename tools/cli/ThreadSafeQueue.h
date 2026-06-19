#pragma once
#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <variant>
#include <optional>

enum class State {
    WaitingForInput,
    Processing,
    ShuttingDown
};


template<typename T>
class ThreadSafeQueue {
private:
    mutable std::mutex          mutex_;
    std::condition_variable     cv_;
    std::queue<T>               queue_;

public:
    ThreadSafeQueue() = default;
    ~ThreadSafeQueue() = default;

    // push by const-ref
    void push(const T& item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(item);
        }
        cv_.notify_one();
    }

    // push by move
    void push(T&& item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
    }

    // blocking pop: waits until an item is available
    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this](){ return !queue_.empty(); });
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    // peek at all items (copy) — use sparingly
    std::vector<T> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> copy = queue_;
        std::vector<T> out;
        while (!copy.empty()) {
            out.push_back(copy.front());
            copy.pop();
        }
        return out;
    }

    // check if empty without popping
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    // get size (approximate)
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    template<class Rep, class Period>
    std::optional<T> pop_for(const std::chrono::duration<Rep,Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this]{ return !queue_.empty(); }))
            return std::nullopt;            // timeout with no data
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

     // non-blocking pop: returns std::nullopt if the queue is empty
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return std::nullopt;
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }
  
};

