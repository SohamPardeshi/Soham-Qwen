# Soham Finetune

## Initialize Virtual Environment

python -m venv qwen3-env
qwen3-env\Scripts\activate.bat


## Download the Facebook Messenger Data
1. Visit https://accountscenter.facebook.com/info_and_permissions
2. Choose to "Export your information". Make sure to select the time range to include as much history as you want.
3. Download each of the files it gives you into the same folder
4. Unzip all of the downloaded files so their contents are in the same place. For me, that was `.\Downloads\fb\`

## Prepare the Dataset

 1. `python data_get.py`

    It will look at `\fb\your_facebook_activity\messages\e2ee_cutover\` and get all of the `.\\{sender}\message_{#}.json` files. It will put them in `.\data\raw` as `message_{sender}_{#}.json`.

2. `python data_clean.py`

    It will look in `.\data\raw` and clean all of the json files and put them in `.\data\clean` in the format `thread_{sender}_{#}.jsonl`

3. `python data_finalize.py`

    It will concatenate all of the jsonl data files in `.\data\clean` into a single large .jsonl file which will act as the dataset.

## Finetune the Model
Run `python run_finetune.py` to go through the training process.

## Interact with the Model
Run `python run_model.py` to prompt the model.