#!/bin/bash
trap "exit" INT


python_cmd=python3

model_class=DNNModel


batch_size=4096

benchmark_py=benchmark_exp_more_complex_prep.py


benchmark_more_py=benchmark_exp_more_complex.py

baseline_py=incremental_updates_base_line.py


incremental_py=incremental_updates_provenance5.py


cd ../src/sensitivity_analysis_SGD/Benchmark_experiments/


period=5

init_iters=60

echo "varied deletion rate::"

dataset_name=MNIST


echo "varied number of samples::"


echo "${python_cmd} ${benchmark_py} 0.001 ${batch_size} 4 [0.8,0.6,0.5,0.5,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] ${model_class} ${dataset_name} 0.2 0.02 1 2"

        ${python_cmd} ${benchmark_py} 0.001 ${batch_size} 4 [0.8,0.6,0.5,0.5,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] ${model_class} ${dataset_name} 0.2 0.02 1 2



for i in 0.00002 0.00005 0.0001 0.0002 0.0005 0.001
#for i in 0.005
do
	echo "deletion rate:: $i"

	for k in 1 2 3 4 5
	do
		echo "random set $k"


		echo "${python_cmd} generate_rand_ids ${i}  ${dataset_name}"

		${python_cmd} generate_rand_ids.py ${i}  ${dataset_name}


#		echo "${python_cmd} ${benchmark_py} 0.001 ${batch_size} 4 [0.8,0.6,0.5,0.5,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] ${model_class} ${dataset_name} 0.2 0.01 1 2"

 #               ${python_cmd} ${benchmark_py} 0.001 ${batch_size} 4 [0.8,0.6,0.5,0.5,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] ${model_class} ${dataset_name} 0.2 0.01 1 2


		for j in 1 2 3 4 5
		do
			echo "repetition $j"

			echo "${python_cmd} ${benchmark_more_py} 0.001 ${batch_size} 8 [0.3,0.3,0.2,0.2,0.1,0.1,0.05,0.05,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] ${model_class} ${dataset_name} 0.2 0.02 1 2"

			${python_cmd} ${benchmark_more_py} 0.001 ${batch_size} 8 [0.3,0.3,0.2,0.2,0.1,0.1,0.05,0.05,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] ${model_class} ${dataset_name} 0.2 0.02 1 2


			echo "baseline::"


			echo "${python_cmd} ${baseline_py} 0"

			${python_cmd} ${baseline_py} 0

	

#		echo "incremental updates::"
#
#		${python_cmd} incremental_updates_provenance3_skipnet.py 20 10

			echo "incremental updates::"

			echo "${python_cmd} ${incremental_py} ${init_iters} ${period}"

			${python_cmd} ${incremental_py} ${init_iters} ${period}
		done

	done

done

<<cmd
echo "varied epochs::"

echo "${python_cmd} generate_rand_ids 0.00002 ${dataset_name}"

${python_cmd} generate_rand_ids.py 0.00002 ${dataset_name}


for i in 4 8 10 12 15 20 25
#for i in 0.005
do

        echo "epoch_num:: $i"

        for j in 1 2 3 4 5 6 7 8 9 10
        do
                echo "repetition $j"

		echo "${python_cmd} ${benchmark_py} 0.001 ${batch_size} $i [0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]  ${model_class} ${dataset_name} 0.2 0.002 1 2"

		${python_cmd} ${benchmark_py} 0.001 ${batch_size} $i [0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]  ${model_class} ${dataset_name} 0.2 0.002 1 2

                echo "baseline::"

		echo "${python_cmd} ${baseline_py} 0"

                ${python_cmd} ${baseline_py} 0



#               echo "incremental updates::"
#
#               ${python_cmd} incremental_updates_provenance3_skipnet.py 20 10

                echo "incremental updates::"

		echo "${python_cmd} ${incremental_py} ${init_iters} ${period}"

                ${python_cmd} ${incremental_py} ${init_iters} ${period}

        done

done







echo "varied l2 norm::"


echo "${python_cmd} generate_rand_ids 0.00002 ${dataset_name}"

${python_cmd} generate_rand_ids.py 0.00002 ${dataset_name}

for i in 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01
#for i in 0.005
do

        echo "l2 norm:: $i"

	for j in 1 2 3 4 5 6 7 8 9 10
	do

		echo "repetition $j"

		echo "${python_cmd} ${benchmark_py} 0.001 ${batch_size} 15 [0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]  DNNModel_skipnet ${dataset_name} 0.2 $i 1 2"

		${python_cmd} ${benchmark_py} 0.001 ${batch_size} 15 [0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]  ${model_class} ${dataset_name} 0.2 $i 1 2

                echo "baseline::"


		echo "${python_cmd} ${baseline_py} 0"

		${python_cmd} ${baseline_py} 0

		echo "incremental updates::"

		echo "${python_cmd} ${incremental_py} ${init_iters} ${period}"

		${python_cmd} ${incremental_py} ${init_iters} ${period}

	done

done
cmd

<<cmd
echo "varied batch size::"

for i in 128 256 1024 4096
#for i in 0.005
do



        echo "batch_size:: $i"

	epoch=$((2*(i/128)))

	echo ${epoch}

        ${python_cmd} benchmark_exp.py 0.001 ${i} ${epoch} DNNModel_single MNIST 0.2 0.05 1 2

        ${python_cmd} incremental_updates_base_line.py 0.00005 0

        ${python_cmd} incremental_updates_provenance3.py 40 15

done





cmd
