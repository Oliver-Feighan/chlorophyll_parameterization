for i in */; 
	do 
	new_file=${i/\//}
	#echo ${i}dscf_${new_file}.in
	sed "s/NAME/${new_file}/g" dscf_template.sub > ${i}dscf_${new_file}.sub
	sed "s/NAME/${new_file}/g" dscf_template.in > ${i}dscf_${new_file}.in
done

