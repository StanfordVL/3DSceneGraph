project_path=/path/to/project 				# system path to the 3D Scene Graph folder
file_path=$project_path/source/framing/pano2rectilinear/equirect2rectilinear.py  # system path to sampling function
gibson_data_path=/path/to/Gibson/data  		# system path to Gibson database model data - used for panoramas
output_dir=/path/to/export/results  		# system path to export results and intermediary files
override=0  								# binary value to override results (if 1 overrides)

ready_models='model1 model2 model3' #replace with names of models to process

for model in $ready_models; do
    printf "Sampling rectilinears for model: $model\n"
    python3 $file_path \
    	--data_path $gibson_data_path \
    	--model $model \
    	--output_dir $output_dir \
    	--override $override
done