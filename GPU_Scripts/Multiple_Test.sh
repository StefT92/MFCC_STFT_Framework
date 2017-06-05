#!/bin/bash
folder='/home/stomassetti/Projects/MFCC_STFT_Framework/PATH_VAR_INIT_files/'
i=1
for file in $(ls $folder); do

	PATH_VAR_INIT_ref=$folder$file
 	echo $PATH_VAR_INIT_ref
	if [ "$i" -eq "1" ]; then
		a=`qsub -v PATH_VAR_INIT_ref=$PATH_VAR_INIT_ref Multiple_Test.pbs`
	else
 		a=`qsub -v PATH_VAR_INIT_ref=$PATH_VAR_INIT_ref Multiple_Test.pbs -W depend=afterok:$a`
	fi
	((i++))
done		
		
	
