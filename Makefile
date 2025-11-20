OS = windows
COMPILER = nvcc

run : objects/main.o objects/global.o objects/Vec3d.o objects/Massive_body.o objects/Test_particle.o objects/init.o objects/tools.o objects/leapfrog.o objects/MVS.o objects/User_forces.o objects/integration.o
	$(COMPILER) -o $@ $^

objects/global.o: sources/global.cpp headers/global.h
	$(COMPILER) -c $< -o $@

objects/Vec3d.o: sources/Vec3d.cu headers/Vec3d.h headers/global.h
	$(COMPILER) -c $< -o $@

objects/Massive_body.o: sources/Massive_body.cpp headers/Massive_body.h headers/global.h
	$(COMPILER) -c $< -o $@

objects/Test_particle.o: sources/Test_particle.cpp headers/Test_particle.h headers/global.h
	$(COMPILER) -c $< -o $@

objects/init.o: sources/init.cpp headers/global.h headers/Vec3d.h headers/Massive_body.h headers/Test_particle.h
	$(COMPILER) -c $< -o $@

objects/tools.o: sources/tools.cu headers/global.h headers/Vec3d.h headers/Massive_body.h headers/Test_particle.h
	$(COMPILER) -c $< -o $@

objects/leapfrog.o: sources/leapfrog.cu headers/global.h headers/Vec3d.h headers/Massive_body.h headers/Test_particle.h headers/tools.h
	$(COMPILER) -c $< -o $@

objects/MVS.o: sources/MVS.cu headers/global.h headers/Vec3d.h headers/Massive_body.h headers/Test_particle.h headers/tools.h
	$(COMPILER) -c $< -o $@

objects/User_forces.o: sources/User_forces.cpp headers/global.h headers/Vec3d.h headers/Massive_body.h
	$(COMPILER) -c $< -o $@

objects/integration.o: sources/integration.cu headers/global.h headers/Vec3d.h headers/Massive_body.h headers/Test_particle.h headers/leapfrog.h headers/MVS.h headers/User_forces.h
	$(COMPILER) -c $< -o $@

objects/main.o: main/main.cu headers/global.h headers/Vec3d.h headers/Massive_body.h headers/Test_particle.h headers/init.h headers/leapfrog.h headers/MVS.h headers/integration.h headers/User_forces.h
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