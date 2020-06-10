#!/bin/bash
trap "exit" INT


python_cmd=python3

model_class=Logistic_regression


#batch_size=16384

#benchmark_py=benchmark_exp_more_complex_prep.py


benchmark_more_py=benchmark_exp_compare_lr.py

baseline_py=incremental_updates_base_line_lr.py


incremental_py=incremental_updates_provenance3_lr.py


incremental_py0=incremental_updates_provenance_lr.py

cd ../src/sensitivity_analysis_SGD/Benchmark_experiments/


period=5

init_iters=20

echo "varied deletion rate::"

dataset_name=MNIST5


echo "varied number of samples::"


#echo "${python_cmd} ${benchmark_py} 0.001 ${batch_size} 4 [0.8,0.6,0.5,0.5,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] ${model_class} ${dataset_name} 0.2 0.02 1 2"

 #       ${python_cmd} ${benchmark_py} 0.001 ${batch_size} 4 [0.8,0.6,0.5,0.5,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] ${model_class} ${dataset_name} 0.2 0.02 1 2



deletion_rate=0.00002


all_deletion_rate=(0.00002 0.00005 0.0001 0.0002 0.0005 0.001)

all_periods=(30 20 15 10 5 2 1)


batch_sizes=(16384)


epoch_nums=(32)


bz=4096

e_num=8

echo "${python_cmd} ${benchmark_more_py} 0.001 ${bz} ${e_num} [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05] ${model_class} ${dataset_name} 0.2 0.005 1 2"

${python_cmd} ${benchmark_more_py} 0.001 ${bz} ${e_num} [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05] ${model_class} ${dataset_name} 0.2 0.005 1 2

#for i in 0.00002 0.00005 0.0001 0.0002 0.0005 0.001
#for i in "${!all_periods[@]}"
for i in "${!all_deletion_rate[@]}"
do

#	period=${all_periods[$i]}

	del_rate=${all_deletion_rate[$i]}

	if ((i==0));
        then
                echo "deletion rate:: ${del_rate}";

                echo "${python_cmd} generate_rand_ids ${del_rate}  ${dataset_name} 1";

                ${python_cmd} generate_rand_ids.py ${del_rate}  ${dataset_name} 1;

        else
                echo "deletion rate:: ${del_rate}";

                echo "${python_cmd} generate_rand_ids ${del_rate}  ${dataset_name} 0";

                ${python_cmd} generate_rand_ids.py ${del_rate}  ${dataset_name} 0;
        fi;





#		echo "${python_cmd} ${benchmark_py} 0.001 ${batch_size} 4 [0.8,0.6,0.5,0.5,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] ${model_class} ${dataset_name} 0.2 0.01 1 2"

 #               ${python_cmd} ${benchmark_py} 0.001 ${batch_size} 4 [0.8,0.6,0.5,0.5,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] ${model_class} ${dataset_name} 0.2 0.01 1 2

# 	for k in "${!batch_sizes[@]}"
#	do
#		bz=${batch_sizes[$k]}
#
#		e_num=${epoch_nums[$k]}
#	
#		echo "batch size:: $bz"
#		
#		for j in 1 2 3 4 5
#		do
#			echo "repetition $j"
#
#			echo "${python_cmd} ${benchmark_more_py} 0.001 ${bz} ${e_num} [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05] ${model_class} ${dataset_name} 0.2 0.005 1 2"
#
#			${python_cmd} ${benchmark_more_py} 0.001 ${bz} ${e_num} [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05] ${model_class} ${dataset_name} 0.2 0.005 1 2


			echo "baseline::"


			echo "${python_cmd} ${baseline_py} 0"

			${python_cmd} ${baseline_py} 0

	


			echo "incremental updates 0::"

                        echo "${python_cmd} ${incremental_py0} 0"

                        ${python_cmd} ${incremental_py0} 0


#		echo "incremental updates::"
#
#		${python_cmd} incremental_updates_provenance3_skipnet.py 20 10

			echo "incremental updates::"

			echo "${python_cmd} ${incremental_py} ${init_iters} ${period}"

			${python_cmd} ${incremental_py} ${init_iters} ${period}
		done

#	done
#
#done












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
