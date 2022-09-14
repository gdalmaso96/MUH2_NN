# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:16:34 2017

@author: schwendimann_p
"""

offset = 0

jobs = open("joblist.txt", "w")

batchsize = []
batchsize.append(50)
batchsize.append(100)
batchsize.append(150)
batchsize.append(200)
batchsize.append(250)
batchsize.append(300)
batchsize.append(350)
batchsize.append(400)
batchsize.append(450)
batchsize.append(500)

num_folds = []
num_folds.append(10)
num_folds.append(5)

commands = []
for b in batchsize:
	for f in num_folds:
		commands.append("python3 /meg/home/dalmaso_g/Surrogates/MUH2_NN/Surrogate_00003.py %d %d\n" %(b, f))

for i in range(len(commands)):
	run = i+offset
	slurm_file = open("/meg/home/dalmaso_g/Surrogates/MUH2_NN/slurm/surrogate00003_%07i.sl" %run,"w")
	jobs.write("sbatch /meg/home/dalmaso_g/Surrogates/MUH2_NN/slurm/surrogate00003_%07i.sl\n" %run)
	
	slurm_file.write("#!/bin/bash\n")
	slurm_file.write("#SBATCH -p all\n")
	slurm_file.write("#SBATCH -o /meg/home/dalmaso_g/Surrogates/MUH2_NN/OUT/surrogate00003_%07i.out \n" %run)
	slurm_file.write("#SBATCH -e /meg/home/dalmaso_g/Surrogates/MUH2_NN/OUT/surrogate00003_%07i.err \n" %run)
	slurm_file.write("#SBATCH --time 3-00:00:00\n\n")
	slurm_file.write("ulimit -c 0\n")
	slurm_file.write("echo Running on: `hostname` \n")
	slurm_file.write("TIMESTART=`date`\n")
	slurm_file.write("echo Start Time: ${TIMESTART}\n")
	slurm_file.write("echo ###################################################################\n")
	slurm_file.write("echo #     Running Environement#\n")
	slurm_file.write("echo ###################################################################\n")
	slurm_file.write("env|sort\n")
	slurm_file.write("echo ###################################################################\n")
	slurm_file.write("echo # End of Running Environement     #\n")
	slurm_file.write("echo ###################################################################\n")
	slurm_file.write(commands[i])
	slurm_file.write("echo Exit status: $?\n")
	slurm_file.write("echo Start Time: ${TIMESTART}\n")
	slurm_file.write("echo Stop Time: `date`\n")
	slurm_file.close()


jobs.close()


