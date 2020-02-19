project_path=/path/to/project           # system path to the 3D Scene Graph folder
code_path=$project_path/source/framing/pano_aggregation/pano_aggreg.py  # system path to panorama aggregation function
gibson_data_path=/path/to/Gibson/data   # system path to Gibson database model data - used for panoramas
model_path=/path/to/export/results      # system path to export results -- also contains the sampled frames and detection results
override=0                              # binary value to override results (if 1 overrides)
detection_folder=mask_rcnn              # name of folder that contains the detections
panoaggreg_folder=pano_aggreg           # name of folder to export panorama aggregation results

ready_models='model1 model2 model3' #replace with names of models to process

for basemodel in $ready_models; do
    model=$model_path/$basemodel
    # check that the detection folder exists
    if [ -d "$model/$detection_folder" ]; then 
        printf "\n\n$basemodel\n"
        output_dir="$model/$panoaggreg_folder"  # the system path to save the panorama aggregation results
        if [ ! -d "$output_dir" ]; then
            mkdir $output_dir
        fi
        data_folder=$gibson_data_path/$basemodel/pano/rgb
        for filename in $model/$detection_folder/*; do
            basefile=$(basename $filename)
            count=`ls -1 $output_dir/$basefile/*.npz 2>/dev/null | wc -l`
            # check if it has been processed (counts npz outputs in the pano folder)
            if [[ "$count" -lt 3 ]] || [[ "$override" -eq 1 ]] ; then
                if [ -f "$model/$detection_folder/$basefile/detection_output.npz" ]; then
                    printf "$basefile\n"                        
                    python3 $code_path \
                        --pano $basefile \
                        --path $model \
                        --output_dir $output_dir \
                        --data_dir $data_folder \
                        --detect_fold $detection_folder \
                        --override $override \
                        --VIS
                fi   
            fi
        done
    fi
done