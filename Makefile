run : objects/main.o objects/global.o objects/Vec3d.o objects/massive_body.o objects/test_particle.o objects/integrators.o
	nvcc -o $@ $^

objects/global.o: sources/global.cpp headers/global.h
	nvcc -c $< -o $@

objects/Vec3d.o: sources/Vec3d.cpp headers/Vec3d.h headers/global.h
	nvcc -c $< -o $@

objects/massive_body.o: sources/massive_body.cpp headers/massive_body.h headers/global.h
	nvcc -c $< -o $@

objects/test_particle.o: sources/test_particle.cpp headers/test_particle.h headers/global.h
	nvcc -c $< -o $@

objects/integrators.o: sources/integrators.cpp headers/test_particle.h headers/global.h headers/Vec3d.h headers/massive_body.h headers/test_particle.h
	nvcc -c $< -o $@

objects/main.o: main/main.cpp headers/global.h headers/Vec3d.h headers/massive_body.h headers/test_particle.h headers/integrators.h
	nvcc -c $< -o $@

all: run

.DEFAULT_GOAL := run


windows_clean:
	del objects\*.o
	del run.exe

windows_clean_results:
	del results\*

windows_clean_all: windows_clean windows_clean_results


linux_clean:
	rm -f objects/*.o
	rm -f run

linux_clean_results:
	rm -f results/*

linux_clean_all: linux_clean linux_clean_results