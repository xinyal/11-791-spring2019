import urllib
import os
import zipfile


# NSP (Nationwide Speech Corpus) ~ scrape from web
def scrape_nsp(todo_files_txt, data_dir):
	with open(todo_files_txt, 'r') as f:
		url_list = [line.strip() for line in f.readlines()]

	for url in url_list:
		# 'https://u.osu.edu/nspcorpus/files/2017/05/at1vowel-1dzjcmi.zip'
		outfile = '{}/{}'.format(data_dir, url.split('/')[-1])
		urllib.urlretrieve(url, outfile)


# from in_zip_dir, unzip (without moving text files) to out_full_dir
def unzip_nsp(in_zip_dir, out_full_dir):
	if not os.path.exists(out_full_dir):
		os.mkdir(out_full_dir)

	for region in os.listdir(in_zip_dir):  # e.g. region_dir=='at'/'mi'
		region_dir = os.path.join(in_zip_dir, region)
		if not os.path.isdir(region_dir) or len(region) != 2:  # ignore 'metadata' dir
			# print region_dir, len(region_dir)
			continue

		for i in range(10):  # 0 through 9
			spkr_dir = '{}{}'.format(region, i)  # e.g. 'at3'
			spkr_path = os.path.join(out_full_dir, spkr_dir)  # en_data/nsp_full/at3
			if not os.path.exists(spkr_path):
				os.mkdir(spkr_path)

			for zip_fname in os.listdir(os.path.join(region_dir)):
				if not zip_fname.endswith('zip') or zip_fname[:3] != spkr_dir:
					continue

				zip_fpath = os.path.join(region_dir, zip_fname)

				zip = zipfile.ZipFile(zip_fpath)
				zip.extractall(spkr_path)

			# e.g. no0: 138MB -> 228MB

			# break

		# break


def create_train_dev_test(file_dir, outfile):
	spkr_idxs = {'train': [2, 3, 4, 7, 8, 9], 'dev': [5, 6], 'test': [0, 1]}
	regions = ['at', 'mi', 'ne', 'no', 'so', 'we']

	for split_type, idxs in spkr_idxs.iteritems():
		outfile_path = '{}.{}'.format(outfile, split_type)
		outfiles_list = []
		for idx in idxs:
			for region in regions:
				spkr = '{}{}'.format(region, idx)  # ex. 'at0', 'mi3'
				spkr_dir = os.path.join(file_dir, spkr)

				for act_type in os.listdir(spkr_dir):
					if act_type.startswith('.'):
						continue

					act_folder = os.path.join(spkr_dir, act_type)

					for wav_file in os.listdir(act_folder):
						if not wav_file.endswith('wav'):
							continue

						outfiles_list.append(os.path.join(act_folder, wav_file))

		with open(outfile_path, 'w') as w:
			for wav_file in outfiles_list:
				w.write(wav_file + '\n')


if __name__ == '__main__':
	# 1. dump web files into 1 folder
	# scrape_nsp('en_data/nsp_todo_download.txt', 'en_data/nsp_zip')

	# 2. manually (from shell)
	# shell ex: mv en_data/nsp/mi* en_data/nsp/mi
	# now ready to be fully zipped and shipped (uploaded somewhere)

	# 3. unzip within sub-directories:
	# zipped dir space = 8.87 GB
	# unzipped dir space = 13.94 GB
	# unzip_nsp('en_data/nsp_zip', 'en_data/nsp_full')

	# 4. convert .aiff to .wav
	# shell: ffmpeg -i {filepath}.aiff {filepath}.wav

	# 5. create list of relative local wav filepaths for train/dev/test
	create_train_dev_test('en_data/nsp_wav', 'en_data/filepaths')
