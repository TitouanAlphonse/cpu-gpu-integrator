run : objects/main.o objects/global.o objects/Vec3d.o objects/massive_body.o objects/test_particle.o objects/integrators.o
	g++ -o $@ $^

objects/global.o: sources/global.cpp headers/global.h
	g++ -Wall -c $< -o $@

objects/Vec3d.o: sources/Vec3d.cpp headers/Vec3d.h headers/global.h
	g++ -Wall -c $< -o $@

objects/massive_body.o: sources/massive_body.cpp headers/massive_body.h headers/global.h
	g++ -Wall -c $< -o $@

objects/test_particle.o: sources/test_particle.cpp headers/test_particle.h headers/global.h
	g++ -Wall -c $< -o $@

objects/integrators.o: sources/integrators.cpp headers/test_particle.h headers/global.h headers/Vec3d.h headers/massive_body.h headers/test_particle.h
	g++ -Wall -c $< -o $@

objects/main.o: main/main.cpp headers/global.h headers/Vec3d.h headers/massive_body.h headers/test_particle.h headers/integrators.h
	g++ -Wall -c $< -o $@

all: run

.DEFAULT_GOAL := run

clean:
	del objects\*.o

clean_wexe: clean
	del run.exe

clean_results:
	del results\*

clean_all: clean_wexe clean_results