






project_path=/path/to/project     # system path to the 3D Scene Graph folder
file_path=$project_path/source/3DSceneGraph/main.py  # system path to 3D Scene Graph generation code
camera_dataset=Taskonomy          # Dataset for camera loading - Gibson or Taskonomy  
camera_path=/path/to/camera/data  # system path to camera pose files
other_path=$data                  # system path to csv files with attributes not analytically computed
room_path=/path/to/room/segmentation    # system path to mesh files of segmented room instances (.obj)
gibson_data_path=/path/to/Gibson/data   # system path to Gibson database model data
result_path=/path/to/export/results     # system path to export results and intermediary files
override=0                              # binary value to override results (if 1 overrides)
mesh_folder="mview_aggreg"              # name of folder to load multiview consistency results
voxel_size=0.1                          # size of voxel (meters)

ready_models="model1 model2 model3"     # replace with names of models to process

count=1
for model in $ready_models; do
    printf "$model : $count, override-->$override, folder-->$mesh_folder\n"
    python $file_path \
        --model $model \
        --model_id $count \
        --mesh_folder $mesh_folder \
        --gibson_data $gibson_data_path \
        --result_path $result_path \
        --room_path $room_path \
        --other_path $other_path \
        --voxel_size $voxel_size \
        --camera_dataset $camera_dataset \
        --camera_path $camera_path \
        --override $override
    let "count=count+1"
done
