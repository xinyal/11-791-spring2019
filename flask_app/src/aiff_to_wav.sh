#!/bin/bash

aiff_dir=en_data/nsp_full
wav_dir=en_data/nsp_wav
for spkr_dir in `ls -1 $aiff_dir`; do
	out_spkr_dir=$wav_dir/$spkr_dir
	mkdir -p $out_spkr_dir
	for activity_dir in `ls -1 $aiff_dir/$spkr_dir`; do
		if [ $activity_dir != "__MACOSX" ]; then
			old_activity_path=$aiff_dir/$spkr_dir/$activity_dir
			activity_path=$out_spkr_dir/$activity_dir
			mkdir $activity_path
			for fname in `ls -1 $old_activity_path`; do
				if [[ "$fname" == *.aiff ]]; then
					ffmpeg -i "$old_activity_path/$fname" "$activity_path/${fname%.aiff}.wav"
				else
					cp "$old_activity_path/$fname" "$activity_path"
				fi
			done

			# break
		fi

	done

	# break # test one
done
