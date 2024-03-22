为了在Ubuntu下使用C++开发MQTT发送端和接收端，你可以使用开源库`mosquitto`，这是一个轻量级的MQTT协议代理，它提供了客户端库，支持多种编程语言，包括C++。下面我将提供一个简单的示例，展示如何创建一个MQTT发布者（发送端）和一个订阅者（接收端）。

### 安装mosquitto和开发库

首先，你需要安装`mosquitto`代理和对应的C语言客户端库：

```bash
sudo apt-get update
sudo apt-get install mosquitto mosquitto-clients libmosquitto-dev
```

### 创建项目

假设我们有一个名为`mqtt_example`的项目，结构如下：

```
mqtt_example/
|-- CMakeLists.txt
|-- publisher.cpp
|-- subscriber.cpp
```

### CMakeLists.txt

这是你的`CMakeLists.txt`文件，它定义了如何构建你的项目：

```cmake
cmake_minimum_required(VERSION 3.5)
project(mqtt_example)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)

# 寻找mosquitto库
find_package(PkgConfig)
pkg_check_modules(MOSQUITTO libmosquitto)

# 包含mosquitto头文件
include_directories(${MOSQUITTO_INCLUDE_DIRS})

# 编译publisher
add_executable(publisher publisher.cpp)
target_link_libraries(publisher ${MOSQUITTO_LIBRARIES})

# 编译subscriber
add_executable(subscriber subscriber.cpp)
target_link_libraries(subscriber ${MOSQUITTO_LIBRARIES})
```

### publisher.cpp

这是一个简单的MQTT发布者示例：

```cpp
#include <mosquitto.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    struct mosquitto *mosq;
    int rc;

    mosquitto_lib_init();

    mosq = mosquitto_new("publisher-test", true, NULL);
    if(mosq == NULL){
        fprintf(stderr, "Error: Out of memory.\n");
        return 1;
    }

    rc = mosquitto_connect(mosq, "localhost", 1883, 60);
    if(rc != 0){
        fprintf(stderr, "Unable to connect (%s).\n", mosquitto_strerror(rc));
        return 1;
    }

    mosquitto_publish(mosq, NULL, "test/topic", strlen("Hello, World!"), "Hello, World!", 0, false);

    mosquitto_disconnect(mosq);
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();

    return 0;
}
```

### subscriber.cpp

这是一个简单的MQTT订阅者示例：

```cpp
#include <mosquitto.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void on_message(struct mosquitto *mosq, void *obj, const struct mosquitto_message *message) {
    printf("Received message: %s\n", (char *) message->payload);
}

int main(int argc, char *argv[]) {
    struct mosquitto *mosq;
    int rc;

    mosquitto_lib_init();

    mosq = mosquitto_new("subscriber-test", true, NULL);
    if(mosq == NULL){
        fprintf(stderr, "Error: Out of memory.\n");
        return 1;
    }

    mosquitto_message_callback_set(mosq, on_message);

    rc = mosquitto_connect(mosq, "localhost", 1883, 60);
    if(rc != 0){
        fprintf(stderr, "Unable to connect (%s).\n", mosquitto_strerror(rc));
        return 1;
    }

    mosquitto_subscribe(mosq, NULL, "test/topic", 0);

    mosquitto_loop_forever(mosq, -1, 1);

    mosquitto_disconnect(mosq);
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();

    return 0;
}
```

### 构建和运行

在你的项目目录（包含`CMakeLists.txt`的目录）中，运行以下命令来构建你的项目：

```bash
mkdir build
cd build
cmake ..
make
```

这将编译生成两个可执行文件：`publisher`和`subscriber`。你可以在一个终端启动`subscriber`
