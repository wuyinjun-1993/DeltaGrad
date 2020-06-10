#!/bin/bash
trap "exit" INT


python_cmd=python3



cd ../src/sensitivity_analysis_SGD/Benchmark_experiments/

<<cmd
${python_cmd} benchmark_exp.py 0.001 256 2 DNNModel_single MNIST 0.2 0.05 1 2

echo "varied number of samples::"



for i in 0.00002 0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01
#for i in 0.005
do

	echo "deletion rate:: $i"

	${python3} incremental_updates_base_line.py ${i} 0

	${python_cmd} incremental_updates_provenance3.py 40 15

done


echo "varied l2 norm::"

for i in 0.001 0.01 0.02 0.05
#for i in 0.005
do

	

        echo "l2 norm:: $i"

	${python_cmd} benchmark_exp.py 0.001 256 2 DNNModel_single MNIST 0.2 0.05 1 2

        ${python3} incremental_updates_base_line.py 0.00005 0

        ${python_cmd} incremental_updates_provenance3.py 40 15

done

cmd
echo "varied batch size::"

for i in 128 256 1024 4096
#for i in 0.005
do



        echo "batch_size:: $i"

	epoch=$((2*(i/128)))

	echo ${epoch}

        ${python_cmd} benchmark_exp.py 0.001 ${i} ${epoch} DNNModel_single MNIST 0.2 0.05 1 2

        ${python3} incremental_updates_base_line.py 0.00005 0

        ${python_cmd} incremental_updates_provenance3.py 40 15

done






