for i in */; 
	do 
	new_file=${i/\//}
	#echo ${i}dscf_${new_file}.in
	#sed "s/NAME/${new_file}/g" dscf_template.sub > ${i}dscf_${new_file}.sub
	#sed "s/NAME/${new_file}/g" dscf_template.in > ${i}dscf_${new_file}.in
	#sed "s/NAME/${new_file}/g" sto_template.sub > ${i}sto_${new_file}.sub
	#sed "s/NAME/${new_file}/g" sto_template.in > ${i}sto_${new_file}.in
	
	sed "s/NAME/${new_file}/g" camb3lyp_template.sub > ${i}camb3lyp_${new_file}.sub
	sed "s/NAME/${new_file}/g" camb3lyp_template.in > ${i}camb3lyp_${new_file}.in

	#sed "s/NAME/${new_file}/g" wB97X_template.sub > ${i}wB97X_${new_file}.sub
	#sed "s/NAME/${new_file}/g" wB97X_template.in > ${i}wB97X_${new_file}.in
	
	#sed "s/NAME/${new_file}/g" BLYP_template.sub > ${i}BLYP_${new_file}.sub
	#sed "s/NAME/${new_file}/g" BLYP_template.in > ${i}BLYP_${new_file}.in
	
	#sed "s/NAME/${new_file}/g" eigdiff_template.sub > ${i}eigdiff_${new_file}.sub
	#sed "s/NAME/${new_file}/g" eigdiff_template.in > ${i}eigdiff_${new_file}.in
done

