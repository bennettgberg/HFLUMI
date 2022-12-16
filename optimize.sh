#!/bin/bash
algo=$2 #"oc"
testname="test${algo}${1}_0"
nproc=8
#xp=0
#yp=6

mkdir latest
mv *.hd5 latest
ls
echo "boutta call offlineHFLumi_v6.py for algo $algo testname $testname nprocesses $nproc"
#python offlineHFLumi_v6.py -d latest/ -m $algo -pc $nproc -o SBR_multi_22hb${algo}${testname} -pf ${algo}params.txt -xp $xp -yp $yp
#python offlineHFLumi_v6.py -d latest/ -m $algo -pc $nproc -o SBR_multi_22hb${algo}${testname} -pf ${algo}params_fullopt.txt
python offlineHFLumi_v6.py -d latest/ -m $algo -pc $nproc -o SBR_multi_${1}${algo}${testname} -pf ${algo}params_${1}.txt
#python offlineHFLumi_v6.py -d hd5_files/latest/ -m $algo -pc $nproc -o SBR_multi_22hb${algo}${testname} -pf ${algo}params.txt -xp $xp -yp $yp
echo "all done :)"
