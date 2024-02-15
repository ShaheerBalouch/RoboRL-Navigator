# Contact Graspnet

```shell
git clone https://github.com/eminsafa/contact_graspnet.git
cd contact_graspnet
conda env create -f contact_graspnet_env.yml
```

### Download Trained Model
Download trained models from [here](https://drive.google.com/file/d/1tQDtYyQv5-QTuLvvPJLhfdZ6tINGBv-L/view?usp=sharing)
and extract files under `external/contact_graspnet/checkpoints`
directory.
It will look like this:
```shell
external
└── contact_graspnet
    └── checkpoints
        └── scene_test_2048_bs3_hor_sigma_001
            ├── checkpoint
            ├── config.yaml
            ...
```

### Run Server
```shell
conda activate contact_graspnet_env
python contact_graspnet/contact_graspnet_server.py
```
