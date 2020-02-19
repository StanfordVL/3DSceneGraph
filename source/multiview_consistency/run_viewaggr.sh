project_path=/path/to/project               # system path to the 3D Scene Graph folder
file_path=$project_path/source/multiview_consistency/start.py  # system path to multiview consistency code
gibson_data_path=/path/to/Gibson/data       # system path to Gibson database model data
output_path=/path/to/export/results         # system path to export results and intermediary files
pano_out=/path/to/pano/aggregation/results  # system path to results of panorama aggregation code
override=0                                  # binary value to override results (if 1 overrides)
pano_type="pano_aggreg"                     # name of folder that contains panorama aggregation results
detect_folder="folder_name_with_detections" # name of folder that contains the detection results
mesh_folder="mview_aggreg"                  # name of folder to export multiview consistency results

ready_models="model1 model2 model3" #replace with names of models to process

for basefile in $ready_models; do
    run=0  # binary value - if 1 the multiview consistency code will run
    model_inp=$pano_out/$basefile/$pano_type  # system path to get panorama aggregation results for this model
    model_out=$output_path/$basefile  # system path to export mview consistency results
    printf "\n\n\n$mesh_folder: $basefile\n"
    # check that there exist panorama aggregation results for this model
    if [ -d "$model_inp/$pano_type" ]; then
        if [ ! -z "$(ls -A $model_inp/$pano_type)" ]; then
            # check if anything has been already computed, to avoid starting from scratch
            if [ ! -d "$model_out/$mesh_folder" ]; then 
                # if nothing has been computed, check that panorama aggregation results exist
                count_old_folders=`find $model_out/$pano_type/p* -maxdepth 0 -type d | wc -l`
                count_pano=`ls -1 $model_out/$detect_type/p*/detection_output.npz 2>/dev/null | wc -l`
                printf "$basefile : $count_old_folders $count_pano\n" 
                if [[ $count_old_folders -eq $count_pano ]]; then 
                    printf "no mesh aggreg folder and all panos are aggregated\n"
                    mkdir $model_out/$mesh_folder
                    run=1
                fi
                run=1
            else
                count=`ls -1 $model_out/$mesh_folder/*.npz 2>/dev/null | wc -l`
                if [[ $count -lt 5 ]]; then
                    printf "not all npz files done\n"
                    run=1
                fi
                if [[ $count -eq 5 ]]; then
                    count_folders=`find $model_out/$mesh_folder/p* -maxdepth 0 -type d | wc -l`
                    count_old_folders=`find $model_out/$pano_type/p* -maxdepth 0 -type d | wc -l`
                    if [[ $count_folders -lt $count_old_folders ]]; then
                        printf "not all mesh2pano projections done\n"
                        run=1
                    fi
                fi  
            fi
        fi
    fi
    if [[ $override -eq 1 ]]; then
        run=1
    fi
    # run the mview consistency code if run == 1
    if [[ $run -eq 1 ]]; then
        printf "\n\n\n$mesh_folder: $basefile\n"
        python3 $file_path \
            --path_model $gibson_data_path \
            --path_out $output_path \
            --path_pano $pano_out \
            --pano_type $pano_type \
            --model $basefile \
            --mesh_folder $mesh_folder \
            --override $override
    fi
done