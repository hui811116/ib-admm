#!/bin/bash
PROGRAM=run.py
EXE=python3
METHOD=alm
penalty_range=(2 32 512)
omega_range=(0 4 16)
nrun=25

for pc in ${penalty_range[*]} ; do
	for om in ${omega_range[*]} ; do
		outfile=result_alm_c${pc}_o${om}
		$EXE $PROGRAM $METHOD $outfile -penalty ${pc} -omega ${om} -v -ntime ${nrun}
		#echo "writing file $outfile"
	done
done