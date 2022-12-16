#!/bin/bash
algo="oc"
testname="test109"
nproc=8
xp=0
yp=6

mkdir latest
mv *.hd5 latest
ls
echo "boutta call offlineHFLumi_v6.py"
python offlineHFLumi_v6.py -d latest/ -m $algo -pc $nproc -o SBR_multi_22hb${algo}${testname} -pf ${algo}params.txt -xp $xp -yp $yp
#python offlineHFLumi_v6.py -d hd5_files/latest/ -m $algo -pc $nproc -o SBR_multi_22hb${algo}${testname} -pf ${algo}params.txt -xp $xp -yp $yp
echo "all done :)"
