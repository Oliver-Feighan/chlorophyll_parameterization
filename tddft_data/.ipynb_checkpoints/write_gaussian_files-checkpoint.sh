for i in step*/;
    do 
        bchla_name=${i/\//}
        echo $bchla_name
        
        xyz_file=${bchla_name}/${bchla_name}.xyz
        
        methods=(CAMB3LYP PBE0 wB97XD BLYP ZINDO)
        
        functionals=(cam-b3lyp PBE1PBE wB97XD blyp)
        
        #write TDDFT files
        for i in ${!functionals[@]};
            do    
                sed -e "s/FUNC/${functionals[$i]}/g" -e "s/NAME/${bchla_name}/g" gaussian_tddft_template.txt;
                sed "1d;2d" ${xyz_file}
                echo
        done;

        #write ZINDO file
        sed "1d;2d" ${xyz_file}
        echo
        
        #write submission script
        g16_string="time g16 < COM > LOG"
        
        sed "s/NAME/${bchla_name}/g" gaussian_submission_template.txt
    
        for i in ${!methods[@]};
            do
                method_string=$(echo ${g16_string} | sed -e "s/COM/${methods[$i]}_${bchla_name}.com/g" -e "s/LOG/${methods[$i]}_${bchla_name}.com/g");
                
                sed "s/${methods[${i}]}/${method_string}/g" gaussian_submission_template.txt
        done

    exit 0
        
done


        