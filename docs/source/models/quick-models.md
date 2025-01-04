<!--
 * @Date: 2024-12-03 04:54:21
 * @LastEditors: caishaofei caishaofei@stu.pku.edu.cn
 * @LastEditTime: 2024-12-03 06:02:49
 * @FilePath: /MineStudio/docs/source/models/quick-models.md
-->

Here is an example that shows how to load the OpenAI's VPT policy in the Minecraft environment. 

```python
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback
from minestudio.models import load_vpt_policy, VPTPolicy

# load the policy from the local model files
policy = load_vpt_policy(
    model_path="/path/to/foundation-model-2x.model", 
    weights_path="/path/to/foundation-model-2x.weights"
).to("cuda")

# or load the policy from the Hugging Face model hub
policy = VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.rl_from_early_game_2x").to("cuda")

policy.eval()

env = MinecraftSim(
    obs_size=(128, 128), 
    callbacks=[RecordCallback(record_path="./output", fps=30, frame_type="pov")]
)
memory = None
obs, info = env.reset()
for i in range(1200):
    action, memory = policy.get_action(obs, memory, input_shape='*')
    obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

```{hint}
In this example, the recorded video will be saved in the `./output` directory. 
```
