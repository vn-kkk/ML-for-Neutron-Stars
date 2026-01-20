# Makefile for compiling Fortran 77 modules using f2py with gfortran

# Compiler and flags
F2PY = f2py
FCOMP = gnu95
FOPT = -O2 -fno-automatic -fno-asynchronous-unwind-tables

# Fortran 77 source files and module names
MODULES = eos tov tov_tide
SOURCES = $(patsubst %, %.f, $(MODULES))
TARGETS = $(patsubst %, %.so, $(MODULES))

# Default target: build all modules
all: $(TARGETS)

%.so: %.f
	$(F2PY) -c $< -m $* --fcompiler=$(FCOMP) --opt="$(FOPT)"

# Clean build artifacts
clean:
	rm -f *.so *.o *.mod

.PHONY: all clean
