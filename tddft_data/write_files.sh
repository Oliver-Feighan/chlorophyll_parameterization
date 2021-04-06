for i in *xyz; 
	do 
	sed "s/NAME/${i/.xyz/}/g" template.in > ${i/.xyz/.in}; 
	sed "s/NAME/${i/.xyz/}/g" template.sub > ${i/.xyz/.sub}; 
done

