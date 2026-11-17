// 示例语料 · 一个最小可读的 epoll 回显服务器（仅用于演示切分与引用，非生产代码）
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

class DemoServer {
public:
    explicit DemoServer(uint16_t port);
    ~DemoServer();

    // 启动：创建监听 fd、eventfd 与 epoll 实例
    bool start();

    // 事件循环：阻塞直到 stop() 被调用或发生致命错误
    void loop();

    // 线程安全：只做一次 8 字节写入，用于唤醒阻塞中的 epoll_wait
    void stop();

private:
    void acceptNewConnections();
    void handleRead(int fd);
    void handleWrite(int fd);
    void closeConnection(int fd);

    uint16_t port_;
    int listenFd_ = -1;
    int epollFd_ = -1;
    int wakeupFd_ = -1;
    bool running_ = false;

    struct Connection {
        std::string outBuffer;
        size_t sent = 0;
    };
    std::unordered_map<int, Connection> connections_;
};
