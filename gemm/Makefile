CC?=gcc
CXX?=g++
VERIFY ?= 0

ifeq ($(VERIFY), 1)
  $(info VERIFYING RESULTS)
endif
CFLAGS+=-I. -DVERIFY=${VERIFY}
CXXFLAGS+=-I. -std=c++11 -DVERIFY=${VERIFY} 
ifeq (${DEBUG}, 1)
  $(info DEBUG ON)
  CFLAGS += -g -pg
  CXXFLAGS += -g -pg
endif
ifdef PRINT
  CFLAGS += -DPRINT=1
  CXXFLAGS += -DPRINT=1
endif
ifdef NUM_REPS
  $(info NUM_REPS = ${NUM_REPS})
  CFLAGS += -DNUM_REPS=${NUM_REPS}
  CXXFLAGS += -DNUM_REPS=${NUM_REPS}
endif
SRC = main.cpp
OBJ = gemm_tester


# GCC compiler
ifeq ($(CC), gcc)
  COFFLOADING = -foffload="-lm" -lm
  C_NO_OFFLOADING = -foffload=disable
  CFLAGS += -O3 -std=c99 -fopenmp $(COFFLOADING) 
  CLINK = gcc
  CLINKFLAGS = -O3 -fopenmp $(COFFLOADING)
endif

# Clang compiler
ifeq ($(CC), clang)
  COFFLOADING = -fopenmp-targets=nvptx64-nvida-cuda
  C_NO_OFFLOADING = 
  CFLAGS += -O3 -std=c99 -fopenmp $(COFFLOADING) -D__STRICT_ANSI__ -D__NO_MATH_INLINES -U__SSE2_MATH__ -U__SSE_MATH__
  CLINK = clang
  CLINKFLAGS = -O3 -fopenmp $(COFFLOADING)
endif

# GCC compiler
ifeq ($(CXX), g++)
  CXXOFFLOADING = -foffload="-lm" -lm
  CXX_NO_OFFLOADING = -foffload=disable
  CXXFLAGS += -O3 -fopenmp $(CXXOFFLOADING)
  CXXLINK = g++
  CXXLINKFLAGS = -O3 -fopenmp $(CXXOFFLOADING)
endif

# Clang compiler
ifeq ($(CXX), clang++)
  CXXOFFLOADING = -fopenmp-targets=nvptx64-nvida-cuda 
  CXX_NO_OFFLOADING = 
  CXXFLAGS += -O3 -fopenmp $(CXXOFFLOADING) -D__STRICT_ANSI__ -D__NO_MATH_INLINES -U__SSE2_MATH__ -U__SSE_MATH__
  CXXLINK = clang++
  CXXLINKFLAGS = -O3 -fopenmp $(CXXOFFLOADING)
endif


all: gemm_tester

gemm_tester: $(SRC)
	$(CXX) -o $@ $(CXXFLAGS) $< 

clean:
	rm gemm_tester
