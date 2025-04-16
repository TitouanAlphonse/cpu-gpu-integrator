COMPILER = nvcc
OS = linux

run : objects/main.o objects/global.o objects/Vec3d.o objects/massive_body.o objects/test_particle.o objects/integrators.o
	$(COMPILER) -o $@ $^

objects/global.o: sources/global.cpp headers/global.h
	$(COMPILER) -c $< -o $@

objects/Vec3d.o: sources/Vec3d.cu headers/Vec3d.h headers/global.h
	$(COMPILER) -c $< -o $@

objects/massive_body.o: sources/massive_body.cpp headers/massive_body.h headers/global.h
	$(COMPILER) -c $< -o $@

objects/test_particle.o: sources/test_particle.cpp headers/test_particle.h headers/global.h
	$(COMPILER) -c $< -o $@

objects/integrators.o: sources/integrators.cu headers/test_particle.h headers/global.h headers/Vec3d.h headers/massive_body.h headers/test_particle.h
	$(COMPILER) -c $< -o $@

objects/main.o: main/main.cu headers/global.h headers/Vec3d.h headers/massive_body.h headers/test_particle.h headers/integrators.h
	$(COMPILER) -c $< -o $@

all: run

.DEFAULT_GOAL := run

clean:
ifeq ($(OS), linux)
	rm -f objects/*.o
	rm -f run
else ifeq ($(OS), windows)
	del objects\*.o
	del run.exe
	del run.exp
	del run.lib
else
	@echo Unsupported OS
endif

clean_results:
ifeq ($(OS), linux)
	rm -f results/*.txt
else ifeq ($(OS), windows)
	del results\*.txt
else
	@echo Unsupported OS
endif

clean_all: clean clean_results