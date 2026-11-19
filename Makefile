# Makefile for Niikoo -> builds niikoo.exe (Windows only)
#
#   make -j8                   optimised build (release, LTO)                   -> niikoo.exe
#   make pgo -j8               profile-guided build (instrumented build, training run, final build)  -> niikoo.exe
#   make verify -j8            slow build that checks every incremental NNUE update  -> niikoo-verify.exe
#   make bench [NET=net.bin]   run "bench" (DEPTH=9 by default)
#   make avx2                  portable build for any AVX2 CPU                  -> niikoo-avx2.exe
#   make clean                 remove all build products
#   make help                  this text
#
# Options (all can be given on the command line, e.g.  make pgo NET=C:/nets/net.bin PGO_DEPTH=13):
#   ARCH=-march=native         default on Windows: tuned for the CPU you build on (run the exe on that same PC)
#   ARCH=-mavx2                runs on any AVX2 CPU (default when cross-compiling from Linux)
#   LTO=0                      switch link-time optimisation off (default 1)
#   STATIC=0                   link dynamically (default 1: the exe has no DLL dependencies)
#   NET=file                   net loaded for "bench" and for the PGO training run (use it: the net code is hot)
#   PGO_DEPTH=12               depth of the PGO training bench
#
# Windows: use the MSYS2 UCRT64 or MINGW64 terminal.
# Linux / GitHub Codespaces (cross-compile): sudo apt install g++-mingw-w64-x86-64-posix gcc-mingw-w64-x86-64-posix

EXE       ?= niikoo.exe
STATIC    ?= 1
MODE      ?= release
LTO       ?= 1
PGO       ?=
NET       ?=
DEPTH     ?= 9
PGO_DEPTH ?= 12

.DEFAULT_GOAL := all

ifeq ($(OS),Windows_NT)
  # native build inside MSYS2
  ARCH ?= -march=native
  RUN  ?=
else
  # Linux: cross-compile with mingw-w64 (the -posix variant is required: std::thread needs POSIX threads)
  ifeq ($(origin CXX),default)
    CXX := x86_64-w64-mingw32-g++-posix
  endif
  ifeq ($(origin CC),default)
    CC := x86_64-w64-mingw32-gcc-posix
  endif
  ARCH ?= -mavx2
  RUN  ?= wine
endif

ifneq ($(filter-out clean help,$(MAKECMDGOALS)),)
  ifeq ($(shell command -v $(CXX) 2>/dev/null),)
    $(error Compiler '$(CXX)' not found. In Codespaces run: sudo apt install g++-mingw-w64-x86-64-posix gcc-mingw-w64-x86-64-posix)
  endif
endif

# Folder that holds main.cpp. Found automatically (here, or one folder down); override with  make SRCDIR=somefolder
SRCDIR ?= $(patsubst %/,%,$(dir $(firstword $(wildcard main.cpp */main.cpp))))
ifeq ($(SRCDIR),)
  SRCDIR := .
endif
vpath %.cpp $(SRCDIR)

# every C++ file of the engine (add new ones here)
SRCS := main.cpp uci.cpp search.cpp eval.cpp nnue.cpp book.cpp play.cpp tb.cpp

# Fathom (Syzygy probing): tbprobe.c includes tbchess.c itself, so only tbprobe.c is compiled
FATHOM := $(firstword $(wildcard $(SRCDIR)/fathom/src/tbprobe.c $(SRCDIR)/fathom/tbprobe.c))

# ---- optimisation level, per mode ----
ifeq ($(MODE),verify)
  OPT := -O2 -DNNUE_VERIFY
  EXE := $(basename $(EXE))-verify.exe
  LTO := 0
else
  OPT := -O3 -DNDEBUG
endif
ifneq ($(LTO),0)
  OPT += -flto
endif

# ---- profile-guided optimisation: PGO=gen builds an instrumented exe, PGO=use builds with the recorded profile ----
ifeq ($(PGO),gen)
  OPT += -fprofile-generate -fprofile-update=prefer-atomic
  LDFLAGS += -fprofile-generate
else ifeq ($(PGO),use)
  # partial-training: code the training run never touched (book, tablebases, pondering) is optimised normally, not for size
  OPT += -fprofile-use -fprofile-correction -fprofile-partial-training -Wno-missing-profile
endif

CXXFLAGS += -std=c++17 $(OPT) $(ARCH) -Wall -pthread -I$(SRCDIR) -I$(SRCDIR)/fathom -I$(SRCDIR)/fathom/src
CFLAGS   += -std=gnu11 $(OPT) $(ARCH) -w -I$(SRCDIR) -I$(SRCDIR)/fathom -I$(SRCDIR)/fathom/src
LDFLAGS  += -pthread
ifeq ($(STATIC),1)
  LDFLAGS += -static
endif
ifeq ($(MODE)$(PGO),release)
  LDFLAGS += -s
endif

# own folder per mode (the PGO phases share one so the profile files are found), never reused from an older build
BUILD := build/windows-$(MODE)$(if $(PGO),-pgo)
OBJS  := $(SRCS:%.cpp=$(BUILD)/%.o)
ifneq ($(FATHOM),)
  OBJS += $(BUILD)/fathom_tbprobe.o
endif

# objects are rebuilt whenever the compiler flags change (ARCH, LTO, PGO ...), not only when a source changes
STAMP := $(BUILD)/.flags
.PHONY: FORCE
FORCE:
$(STAMP): FORCE
	@mkdir -p $(BUILD)
	@printf '%s\n' '$(CXX) $(CXXFLAGS) | $(CC) $(CFLAGS)' | cmp -s - $@ || printf '%s\n' '$(CXX) $(CXXFLAGS) | $(CC) $(CFLAGS)' > $@

.PHONY: all verify avx2 pgo bench clean help
all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)
	@echo "built $@"

$(BUILD)/%.o: %.cpp $(STAMP)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/fathom_tbprobe.o: $(FATHOM) $(STAMP)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

verify:
	@$(MAKE) MODE=verify

avx2:
	@$(MAKE) ARCH=-mavx2 EXE=niikoo-avx2.exe

# UCI input for bench / the PGO training run: load the net first (if NET is given), then bench at depth $(1)
define BENCHIN
$(if $(NET),setoption name EvalFile value $(NET)\n)bench $(1)\nquit\n
endef

bench: $(EXE)
	@printf '$(call BENCHIN,$(DEPTH))' | $(RUN) ./$(EXE)

PGO_EXE := niikoo-pgogen.exe
pgo:
	@echo "== PGO 1/3: instrumented build"
	@rm -rf build/windows-$(MODE)-pgo
	@$(MAKE) PGO=gen EXE=$(PGO_EXE)
	@echo "== PGO 2/3: training run (bench $(PGO_DEPTH)$(if $(NET), with $(NET), WITHOUT a net: pass NET=... for a better profile))"
	@printf '$(call BENCHIN,$(PGO_DEPTH))' | $(RUN) ./$(PGO_EXE) > /dev/null
	@echo "== PGO 3/3: optimised build with the profile"
	@rm -f build/windows-$(MODE)-pgo/*.o build/windows-$(MODE)-pgo/.flags
	@$(MAKE) PGO=use
	@rm -f $(PGO_EXE)

clean:
	rm -rf build niikoo.exe niikoo-verify.exe niikoo-avx2.exe $(PGO_EXE)

help:
	@sed -n '2,/^# Windows: use/p' Makefile | sed 's/^# \{0,1\}//'

-include $(OBJS:.o=.d)