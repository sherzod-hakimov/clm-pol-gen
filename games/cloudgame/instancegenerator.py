
import sys
import json

#sys.path.append("/Users/antonia/Documents/Unizeug/2023_SoSe_IM/clembench")

from clemgame.clemgame import GameInstanceGenerator

# set the name of the game in the script, as you named the directory
# this name will be used everywhere, including in the table of results
GAME_NAME = 'cloudgame'

class CloudGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        # always do this to initialise GameInstanceGenerator
        super().__init__(GAME_NAME)
    def on_generate(self):

        player_a_prompt_header = self.load_template(f"resources/prompt.template")

        split_path = 'resources/train_1.jsonl'

        dataset = {}
        with open(split_path, "r") as file:
            for l in file.readlines():
                data = json.loads(l)
                dataset[data['id']] = data
                z = 1

        file_path = 'resources/train_iblip_flant5xxl.jsonl'

        experiments = ['prompt_one', 'prompt_two', 'prompt_three', 'prompt_four']

        experiments = ['prompt_one']

        instance_counter = 0

        for experiment in experiments:

            instance_counter = 0

            experiment_name = self.add_experiment('policy_with_iblip_flan_xxl_' + experiment)
            game_counter = 0

            with open(file_path, "r") as file:
                for l in file.readlines():
                    data = json.loads(l)

                    label = dataset[data['id']]['label']
                    if label != '1':
                        continue

                    prompt = player_a_prompt_header.replace('$TEXT',
                                                            dataset[data['id']]['text'])

                    game_instance = self.add_game_instance(experiment_name, game_counter)

                    game_instance["prompt"] = prompt
                    game_instance["image"] = "games/cloudgame/resources/images/" + data['id']
                    game_instance["id"] = data['id']

                    instance_counter += 1
            print(instance_counter)

if __name__ == '__main__':
    # always call this, which will actually generate and save the JSON file
    CloudGameInstanceGenerator().generate()

