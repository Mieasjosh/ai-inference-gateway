CXX ?= g++

DEBUG ?= 1
ifeq ($(DEBUG), 1)
    CXXFLAGS += -g
else
    CXXFLAGS += -O2
endif

# AI Inference Gateway
TARGET = ai_gateway

SRCS = main.cpp \
       webserver.cpp \
       config.cpp \
       ./timer/lst_timer.cpp \
       ./http/http_conn.cpp \
       ./log/log.cpp \
       ./scheduler/batch_scheduler.cpp \
       ./engine/mock_engine.cpp \
       ./engine/onnx_engine.cpp \
       ./memory/buffer_pool.cpp

# --- ONNX Runtime 配置（阶段三）---
# 已启用真实推理：onnxruntime-linux-x64-1.19.2
ONNXRUNTIME_DIR = ./onnxruntime-linux-x64-1.19.2
CXXFLAGS += -DHAS_ONNXRUNTIME
CXXFLAGS += -I$(ONNXRUNTIME_DIR)/include
LDFLAGS  += -L$(ONNXRUNTIME_DIR)/lib -lonnxruntime

# 注意：运行前需设置 LD_LIBRARY_PATH 或 ldconfig 注册 libonnxruntime.so

$(TARGET): $(SRCS)
	$(CXX) -o $(TARGET) $^ $(CXXFLAGS) $(LDFLAGS) -lpthread

clean:
	rm -f $(TARGET)
