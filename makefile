dsdesc.out: dsdesc.cpp
	mpic++ -O3 dsdesc.cpp -o dsdesc.out

clean:
	rm dsdesc.out

run:
	sh run.sh
