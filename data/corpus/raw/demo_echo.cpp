// 示例语料 · demo_echo.cpp（仅用于演示切分与引用，非生产代码）
#include "demo_echo.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace {
constexpr int kMaxEvents = 64;
constexpr size_t kReadBufferSize = 4096;
}  // namespace

DemoServer::DemoServer(uint16_t port) : port_(port) {}

DemoServer::~DemoServer() {
    if (listenFd_ != -1) ::close(listenFd_);
    if (epollFd_ != -1) ::close(epollFd_);
    if (wakeupFd_ != -1) ::close(wakeupFd_);
}

bool DemoServer::start() {
    listenFd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (listenFd_ == -1) return false;

    int reuse = 1;
    ::setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port_);
    if (::bind(listenFd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == -1) return false;
    if (::listen(listenFd_, SOMAXCONN) == -1) return false;

    epollFd_ = ::epoll_create1(EPOLL_CLOEXEC);
    wakeupFd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);

    // 监听 fd 用边沿触发：一次事件要把 accept 队列抽干，否则新连接会滞留
    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLET;
    ev.data.fd = listenFd_;
    ::epoll_ctl(epollFd_, EPOLL_CTL_ADD, listenFd_, &ev);

    // 唤醒 fd 不用边沿触发，读走 8 字节即清空计数
    epoll_event wake{};
    wake.events = EPOLLIN;
    wake.data.fd = wakeupFd_;
    ::epoll_ctl(epollFd_, EPOLL_CTL_ADD, wakeupFd_, &wake);

    running_ = true;
    return true;
}

void DemoServer::loop() {
    epoll_event events[kMaxEvents];

    while (running_) {
        int ready = ::epoll_wait(epollFd_, events, kMaxEvents, -1);
        if (ready == -1) {
            if (errno == EINTR) continue;
            break;
        }

        for (int i = 0; i < ready && running_; ++i) {
            int fd = events[i].data.fd;
            if (fd == wakeupFd_) {
                uint64_t drained = 0;
                ::read(wakeupFd_, &drained, sizeof(drained));
                continue;
            }
            if (fd == listenFd_) {
                acceptNewConnections();
                continue;
            }
            if (events[i].events & (EPOLLHUP | EPOLLERR)) {
                closeConnection(fd);
            } else if (events[i].events & EPOLLOUT) {
                handleWrite(fd);
            } else if (events[i].events & EPOLLIN) {
                handleRead(fd);
            }
        }
    }
}

void DemoServer::stop() {
    running_ = false;
    uint64_t one = 1;
    ssize_t n = ::write(wakeupFd_, &one, sizeof(one));
    (void)n;
}

void DemoServer::acceptNewConnections() {
    while (true) {
        int fd = ::accept4(listenFd_, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (fd == -1) {
            // 边沿触发下必须循环 accept 直到 EAGAIN，否则会漏掉连接
            if (errno == EAGAIN || errno == EWOULDBLOCK) return;
            return;
        }

        epoll_event ev{};
        ev.events = EPOLLIN | EPOLLOUT | EPOLLET;
        ev.data.fd = fd;
        if (::epoll_ctl(epollFd_, EPOLL_CTL_ADD, fd, &ev) == -1) {
            ::close(fd);
            continue;
        }
        connections_[fd] = Connection{};
    }
}

void DemoServer::handleRead(int fd) {
    char buffer[kReadBufferSize];

    while (true) {
        ssize_t n = ::read(fd, buffer, sizeof(buffer));
        if (n > 0) {
            Connection& conn = connections_[fd];
            conn.outBuffer.append(buffer, static_cast<size_t>(n));
            handleWrite(fd);
            continue;
        }
        if (n == 0) {
            // 对端关闭：先把未发完的数据丢掉，再关闭
            closeConnection(fd);
            return;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) return;
        closeConnection(fd);
        return;
    }
}

void DemoServer::handleWrite(int fd) {
    auto it = connections_.find(fd);
    if (it == connections_.end()) return;

    Connection& conn = it->second;
    while (conn.sent < conn.outBuffer.size()) {
        ssize_t n = ::send(fd, conn.outBuffer.data() + conn.sent,
                           conn.outBuffer.size() - conn.sent, MSG_NOSIGNAL);
        if (n > 0) {
            conn.sent += static_cast<size_t>(n);
            continue;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // 发不出去就等 EPOLLOUT，由事件循环再调回来
            return;
        }
        closeConnection(fd);
        return;
    }

    conn.outBuffer.clear();
    conn.sent = 0;
}

void DemoServer::closeConnection(int fd) {
    ::epoll_ctl(epollFd_, EPOLL_CTL_DEL, fd, nullptr);
    ::close(fd);
    connections_.erase(fd);
}
