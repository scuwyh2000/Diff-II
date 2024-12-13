## get inversion pool
python get_inversion.py \
    --datasets='aircraft' \
    --shot='5shot' \

## inversion interpolation
python interpolation_le2.py \
    --strength=0.0 \
    --datasets='aircraft' \
    --shot='5shot' \

## train classifier
# car
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=128 \
    --batch_size=256 \
    --lr=0.1 \
    --size=224 \
    --seed=2020 \
    --syn_p=0 \
    --resize=256 \
    --syn_dir='syn/car/5shot/ours_0.3_5.0' \
    --datasets='car' \
    --num_class=196 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=128 \
    --batch_size=256 \
    --lr=0.1 \
    --size=224 \
    --seed=2020 \
    --syn_p=0 \
    --resize=256 \
    --syn_dir='syn/car/10shot/ours_0.1_5.0' \
    --datasets='car' \
    --num_class=196 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="vit_b_16" \
    --epochs=100 \
    --batch_size=32 \
    --lr=1e-3 \
    --size=384 \
    --seed=2020 \
    --syn_p=0 \
    --resize=440 \
    --syn_dir='syn/car/5shot/ours_0.3_5.0' \
    --datasets='car' \
    --num_class=196 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="vit_b_16" \
    --epochs=100 \
    --batch_size=32 \
    --lr=1e-3 \
    --size=384 \
    --seed=2020 \
    --syn_p=0 \
    --resize=440 \
    --syn_dir='syn/car/10shot/ours_0.1_5.0' \
    --datasets='car' \
    --num_class=196 \

# pet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=128 \
    --batch_size=256 \
    --lr=0.01 \
    --size=224 \
    --seed=2020 \
    --syn_p=0 \
    --resize=256 \
    --syn_dir='syn/pet/5shot/ours_0.3_5.0' \
    --datasets='pet' \
    --num_class=37 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=128 \
    --batch_size=256 \
    --lr=0.01 \
    --size=224 \
    --seed=2020 \
    --syn_p=0 \
    --resize=256 \
    --syn_dir='syn/pet/10shot/ours_0.1_5.0' \
    --datasets='pet' \
    --num_class=37 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="vit_b_16" \
    --epochs=100 \
    --batch_size=32 \
    --lr=1e-4 \
    --size=384 \
    --seed=2020 \
    --syn_p=0 \
    --resize=440 \
    --syn_dir='syn/pet/5shot/ours_0.3_5.0' \
    --datasets='pet' \
    --num_class=37 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="vit_b_16" \
    --epochs=100 \
    --batch_size=32 \
    --lr=1e-4 \
    --size=384 \
    --seed=2020 \
    --syn_p=0 \
    --resize=440 \
    --syn_dir='syn/pet/10shot/ours_0.1_5.0' \
    --datasets='pet' \
    --num_class=37 \

# aircraft
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=128 \
    --batch_size=256 \
    --lr=0.1 \
    --size=224 \
    --seed=2020 \
    --syn_p=0 \
    --resize=256 \
    --syn_dir='syn/aircraft/5shot/ours_1.0_5.0' \
    --datasets='aircraft' \
    --num_class=100 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=128 \
    --batch_size=256 \
    --lr=0.1 \
    --size=224 \
    --seed=2020 \
    --syn_p=0 \
    --resize=256 \
    --syn_dir='syn/aircraft/10shot/ours_1.0_5.0' \
    --datasets='aircraft' \
    --num_class=100 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="vit_b_16" \
    --epochs=100 \
    --batch_size=32 \
    --lr=1e-3 \
    --size=384 \
    --seed=2020 \
    --syn_p=0 \
    --resize=440 \
    --syn_dir='syn/aircraft/5shot/ours_1.0_5.0' \
    --datasets='aircraft' \
    --num_class=100 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="vit_b_16" \
    --epochs=100 \
    --batch_size=32 \
    --lr=1e-3 \
    --size=384 \
    --seed=2020 \
    --syn_p=0 \
    --resize=440 \
    --syn_dir='syn/aircraft/10shot/ours_1.0_5.0' \
    --datasets='aircraft' \
    --num_class=100 \



# cub
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=128 \
    --batch_size=256 \
    --lr=0.05 \
    --size=224 \
    --seed=2020 \
    --syn_p=0 \
    --resize=256 \
    --syn_dir='syn/cub/5shot/ours_1.0_5.0' \
    --datasets='cub' \
    --num_class=200 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=128 \
    --batch_size=256 \
    --lr=0.05 \
    --size=224 \
    --seed=2020 \
    --syn_p=0 \
    --resize=256 \
    --syn_dir='syn/cub/10shot/ours_1.0_5.0' \
    --datasets='cub' \
    --num_class=200 \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="vit_b_16" \
    --epochs=100 \
    --batch_size=32 \
    --lr=1e-3 \
    --size=384 \
    --seed=2020 \
    --syn_p=0 \
    --resize=440 \
    --syn_dir='syn/cub/5shot/ours_1.0_5.0' \
    --datasets='cub' \
    --num_class=200 \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="vit_b_16" \
    --epochs=100 \
    --batch_size=32 \
    --lr=1e-3 \
    --size=384 \
    --seed=2020 \
    --syn_p=0 \
    --resize=440 \
    --syn_dir='syn/cub/10shot/ours_1.0_5.0' \
    --datasets='cub' \
    --num_class=200 \


## train classifier on long-tail datasets
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier_imb.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=200 \
    --batch_size=128 \
    --size=224 \
    --seed=2020 \
    --syn_p=0.5 \
    --resize=256 \
    --syn_dir='syn/imb_cub_0.1/data' \
    --datasets='cub' \
    --num_class=200 \
    --imb_f=0.1 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier_imb.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=200 \
    --batch_size=128 \
    --size=224 \
    --seed=2020 \
    --syn_p=0.5 \
    --resize=256 \
    --syn_dir='syn/imb_cub_0.05/data' \
    --datasets='cub' \
    --num_class=200 \
    --imb_f=0.05 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier_imb.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=200 \
    --batch_size=128 \
    --size=224 \
    --seed=2020 \
    --syn_p=0.5 \
    --resize=256 \
    --syn_dir='syn/imb_cub_0.01/data' \
    --datasets='cub' \
    --num_class=200 \
    --imb_f=0.01 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier_imb.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=200 \
    --batch_size=128 \
    --size=224 \
    --seed=2020 \
    --syn_p=0.5 \
    --resize=256 \
    --syn_dir='syn/imb_flower_0.1/data' \
    --datasets='flower' \
    --num_class=102 \
    --imb_f=0.1 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier_imb.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=200 \
    --batch_size=128 \
    --size=224 \
    --seed=2020 \
    --syn_p=0.5 \
    --resize=256 \
    --syn_dir='syn/imb_flower_0.05/data' \
    --datasets='flower' \
    --num_class=102 \
    --imb_f=0.05 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier_imb.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=200 \
    --batch_size=128 \
    --size=224 \
    --seed=2020 \
    --syn_p=0.5 \
    --resize=256 \
    --syn_dir='syn/imb_flower_0.01/data' \
    --datasets='flower' \
    --num_class=102 \
    --imb_f=0.01 \


## train classifier and test on ood datasets
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=128 \
    --batch_size=256 \
    --lr=0.05 \
    --size=224 \
    --seed=2020 \
    --syn_p=0.5 \
    --resize=256 \
    --syn_dir='syn/cub/5shot/ours_1.0_5.0' \
    --datasets='cub' \
    --num_class=100 \
    --val_dataset='datasets/waterbird/test00.txt' \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=128 \
    --batch_size=256 \
    --lr=0.05 \
    --size=224 \
    --seed=2020 \
    --syn_p=0.5 \
    --resize=256 \
    --syn_dir='syn/cub/5shot/ours_1.0_5.0' \
    --datasets='cub' \
    --num_class=100 \
    --val_dataset='datasets/waterbird/test01.txt' \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=128 \
    --batch_size=256 \
    --lr=0.05 \
    --size=224 \
    --seed=2020 \
    --syn_p=0.5 \
    --resize=256 \
    --syn_dir='syn/cub/5shot/ours_1.0_5.0' \
    --datasets='cub' \
    --num_class=100 \
    --val_dataset='datasets/waterbird/test10.txt' \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_classifier.py \
    --pretrained \
    --arch="resnet50" \
    --epochs=128 \
    --batch_size=256 \
    --lr=0.05 \
    --size=224 \
    --seed=2020 \
    --syn_p=0.5 \
    --resize=256 \
    --syn_dir='syn/cub/5shot/ours_1.0_5.0' \
    --datasets='cub' \
    --num_class=100 \
    --val_dataset='datasets/waterbird/test11.txt' \
