echo '{'

for i in */*_dscf.out;
	do

	#get file name
	arr_file_name=(${i/\// })
	file_name=${arr_file_name[0]}

	#read excitation energy
	excitation_line=$(grep "Excitation energy (in" ${i})
	excitation_energy=$(echo ${excitation_line} | awk '{ print $5 }')
	
	#read transition dipole
	transition_dipole_line=$(grep "Transition dipole" ${i})
	transition_dipole=$(echo $transition_dipole_line | awk '{ print $3 ", " $4 ", " $5 }')

	echo \"$file_name\" ": { \"energy\" " : $excitation_energy ", \"transition_dipole\" : [" $transition_dipole "]" '},'

done

echo '}'