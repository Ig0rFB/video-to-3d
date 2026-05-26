# MuSHRoom training

**Full guide:** [docs/MUSHROOM.md](../../docs/MUSHROOM.md)

**Cloud setup:** [docs/CLOUD_GPU.md](../../docs/CLOUD_GPU.md)

```bash
./scripts/setup_cloud.sh
uv run --no-sync python input/download_mushroom.py --room coffee_room
uv run --no-sync python 03_train_gaussian.py --mushroom input/MuSHRoom/room_datasets/coffee_room
```
