#!/bin/bash

#SBATCH --job-name=HJ_pw1
#SBATCH --output=log.train_pgd_waymo_n1.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=64000
#SBATCH --mail-user=hyejun.lee@sjsu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --partition=gpu

export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
echo ':: Start ::'
source ~/.bashrc
conda activate cmpe295
python tools/train.py configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_n1.py --cfg-options load_from="global_weight_pgd.pth"

# Source file with path
src_file="work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_n1/epoch_5.pth"

# Extract the directory path from the source file
src_dir=$(dirname "$src_file")

# Extract the base name and extension
base_name=$(basename "$src_file" .pth)
extension="${src_file##*.}"

# Initialize the counter for duplicate file names
counter=1

# Initialize the new file name with the path included
new_file="${src_dir}/${base_name}.${extension}"

# Loop to find a new file name if a file with the same name exists
while [ -f "$new_file" ]; do
    new_file="${src_dir}/${base_name}_${counter}.${extension}"
    ((counter++))
done

# Copy the file with the new file name
cp "$src_file" "$new_file"

echo "File copied as: $new_file"
echo ':: End ::'
