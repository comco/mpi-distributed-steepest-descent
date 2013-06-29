dsdesc.out: dsdesc.cpp
	mpic++ -Ofast dsdesc.cpp -o dsdesc.out

clean:
	rm dsdesc.out

run:
	sh run.sh
