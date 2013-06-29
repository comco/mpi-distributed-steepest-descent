for p in 1 2 4 8
do
    for n in 16 32 64 128 256
    do
        echo "processors: $p, size: $n"
        mpirun -np $p ./dsdesc.out $n > times/p_$p.n_$n.txt
    done
done
