OS = windows
COMPILER = nvcc

run : objects/main.o objects/global.o objects/Vec3d.o objects/massive_body.o objects/test_particle.o objects/init.o objects/leapfrog.o objects/integration.o
	$(COMPILER) -o $@ $^

objects/global.o: sources/global.cpp headers/global.h
	$(COMPILER) -c $< -o $@

objects/Vec3d.o: sources/Vec3d.cu headers/Vec3d.h headers/global.h
	$(COMPILER) -c $< -o $@

objects/massive_body.o: sources/massive_body.cpp headers/massive_body.h headers/global.h
	$(COMPILER) -c $< -o $@

objects/test_particle.o: sources/test_particle.cpp headers/test_particle.h headers/global.h
	$(COMPILER) -c $< -o $@

objects/init.o: sources/init.cpp headers/global.h headers/Vec3d.h headers/massive_body.h headers/test_particle.h
	$(COMPILER) -c $< -o $@

objects/leapfrog.o: sources/leapfrog.cu headers/global.h headers/Vec3d.h headers/massive_body.h headers/test_particle.h
	$(COMPILER) -c $< -o $@

objects/integration.o: sources/integration.cu headers/global.h headers/Vec3d.h headers/massive_body.h headers/test_particle.h
	$(COMPILER) -c $< -o $@

objects/main.o: main/main.cu headers/global.h headers/Vec3d.h headers/massive_body.h headers/test_particle.h headers/init.h headers/leapfrog.h headers/integration.h
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

clean_outputs:
ifeq ($(OS), linux)
	rm -f outputs/*.txt
else ifeq ($(OS), windows)
	del outputs\*.txt
else
	@echo Unsupported OS
endif

clean_all: clean clean_outputs