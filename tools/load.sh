project_path=/path/to/project                   # system path to the 3D Scene Graph folder
file_path=$project_path/tools/load.py           # system path to loading function
verified=0                                      # 0 or 1 depending on whether to load only automated or verified results
visualize=0                                     # 0 or 1 depending on whether to export or not wavefront files with segmentation
data_path=$project_path/data                    # system path to 3D Scene Graph results
palette_path=$project_path/tools/palette.txt    # system path to color palette (for visualization purposes)
gibson_mesh_path=/path/to/Gibson/database       # system path to Gibson database model data
export_viz_path=$project_path/visuals           # system path to export wavefront files (if visualize is set to 1)

# list of models in the tiny Gibson split
models='Allensville Beechwood Benevolence Coffeen Collierville Corozal Cosmos Darden Forkland Hanson Hiteman Ihlen Klickitat Lakeville Leonardo Lindenwood Markleeville Marstons McDade Merom Mifflinburg Muleshoe Newfields Noxapater Onaga Pinesdale Pomaria Ranchester Shelbyville Stockman Tolstoy Uvalda Wainscott Wiconisco Woodbine'

#iterate over models
for model in $models; do
    python $file_path --model $model \
                    --verified $verified \
                    --visualize $visualize \
                    --data_path $data_path \
                    --palette_path $palette_path \
                    --gibson_mesh_path $gibson_mesh_path \
                    --export_viz_path $export_viz_path
done