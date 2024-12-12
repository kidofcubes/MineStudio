### YAML Tutorial for Task Configuration in MineStudio

The YAML file defines a task for the MineStudio benchmark. This tutorial explains its components and usage within the framework.

---

#### Sample YAML Structure

```yaml
custom_init_commands: 
- /give @s minecraft:water_bucket 3
- /give @s minecraft:stone 64
- /give @s minecraft:dirt 64
- /give @s minecraft:shovel{Enchantments:[{id:"minecraft:efficiency",lvl:1}]} 1
defaults:
- base
- _self_
text: Build a waterfall in your Minecraft world.
```

---

### Key Elements of the YAML File

1. **`custom_init_commands`**:
   - Specifies commands to initialize the Minecraft environment for the task.
   - Examples:
     - `/give @s minecraft:water_bucket 3`: Gives the agent three water buckets.
     - `/give @s minecraft:stone 64`: Provides a stack of stone blocks.
   - These commands ensure the agent has the necessary tools and resources to perform the task.

2. **`defaults`**:
   - Refers to baseline configurations to include in the task. 
   - `base`: Refers to a predefined set of configurations.
   - `_self_`: Ensures that the YAML file includes its custom specifications.

3. **`text`**:
   - Provides a natural language description of the task.
   - Example: `"Build a waterfall in your Minecraft world."`

---

### Using YAML in the Framework 

1. **Location**:
   - Save the YAML file in the `task_configs/` directory under the appropriate difficulty subdirectory (e.g., `simple/` or `hard/`).

2. **Loading the YAML**:
   - Use the provided `read_conf.py` script to load and parse YAML files.
   - Example usage:
     ```python
     from utility.read_conf import convert_yaml_to_callbacks

     commands_callback, task_callback = convert_yaml_to_callbacks("task_configs/simple/build_waterfall.yaml")
     ```

3. **Task Execution**:
   - The parsed callbacks are passed to the simulation during environment initialization:
     ```python
     env = MinecraftSim(callbacks=[
         CommandsCallback(commands_callback),
         TaskCallback(task_callback)
     ])
     ```

4. **Test the Task**:
   - Run the `test.py` script:
     ```bash
     python test.py --task-config task_configs/simple/build_waterfall.yaml
     ```
