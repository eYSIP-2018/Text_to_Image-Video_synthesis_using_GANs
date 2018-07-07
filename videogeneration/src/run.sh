
#put in the paramter values as you want them , recommended values are these

image_batch=32 \
video_batch=32 \
noise_sigma=0.1 \
print_every=100 \
every_nth=2 \
dim_z_content=50\
dim_z_motion=10\
dim_z_category=6 \


#this shell script runs the train function with all the hyper paramaters
# make sure the logs/actions or logs/shapes contains the dataset
python train.py  \
    --image_batch ${image_batch} \
    --video_batch ${video_batch} \
    --use_infogan \
    --use_noise \
    --noise_sigma ${noise_sigma} \
    --image_discriminator PatchImageDiscriminator \
    --video_discriminator CategoricalVideoDiscriminator \
    --print_every ${print_every} \
    --every_nth ${every_nth} \
    --dim_z_content ${dim_z_content} \
    --dim_z_motion ${dim_z_motion} \
    --dim_z_category  ${dim_z_category}\
    ../data/actions ../logs/actions