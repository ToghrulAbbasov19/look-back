# Third-party notices

The code in this repository is under the MIT license (`LICENSE`), except for the files below.

| Files | Origin | License |
|---|---|---|
| `chrome_dino/chrome_dino_game/envs.py` | [aome510/chrome-dino-game-rl](https://github.com/aome510/chrome-dino-game-rl) | MIT, declared in its `pyproject.toml` (`LICENSES/MIT-chrome-dino-game-rl.txt`) |
| `chrome_dino/diffusion/{config,data,model,sampling,train_world_model,utils}.py` | [arnaudstiegler/gameNgen-repro](https://github.com/arnaudstiegler/gameNgen-repro); `train_world_model.py` builds on a Hugging Face diffusers training script | Apache-2.0 (`LICENSES/Apache-2.0.txt`) |
| `chrome_dino/chrome_dino_game/assets/*.png` | Chrome Dino sprites from Chromium, via [dhhruv/Chrome-Dino-Runner](https://github.com/dhhruv/Chrome-Dino-Runner) | BSD-3-Clause (`LICENSES/BSD-3-Clause-Chromium.txt`); the intermediate repository states no license |

The adapted files were modified for this project, and each carries a one-line notice naming its origin.

Dino weights and datasets are hosted separately (see `README.md`):

| Weights or data | License |
|---|---|
| Released VAE, fine-tuned from the Stable Diffusion v1-4 VAE (`CompVis/stable-diffusion-v1-4`) | CreativeML OpenRAIL-M |
| Released diffusion world model, reward model, policies and datasets | MIT, as this repository |

World-model evaluation uses the [counterfactual benchmark](https://github.com/gulnazaki/counterfactual-benchmark)
(MIT) as an external checkout; it is not included here.
