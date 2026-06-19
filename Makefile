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
       ./memory/buffer_pool.cpp

$(TARGET): $(SRCS)
	$(CXX) -o $(TARGET) $^ $(CXXFLAGS) -lpthread

clean:
	rm -f $(TARGET)
