# This file should be in the same dir as the uncompressed data

mkdir -p ./MFCC
for f in `ls */*/*.wav`; do
    python toMFCC.py $f ./MFCC
done 
