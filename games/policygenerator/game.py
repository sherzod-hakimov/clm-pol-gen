from typing import Dict, List

from clemgame.clemgame import Player


class Instruction:

    def __init__(self):
        self.user_messages = []
        self.system_messages = []

    def add_user_message(self, message):
        self.user_messages.append(message)

    def add_system_message(self, message):
        self.system_messages.append(message)

    def convert_to_query_messages(self):
        messages = []
        messages.append({"role": "system", "content": ""})
        for i in range(0, len(self.user_messages)):
            messages.append({"role": "user", "content": self.user_messages[i]})

            if i < len(self.system_messages):
                messages.append({"role": "assistant", "content": self.system_messages[i]})

        return messages

    def serialize(self):
        output = []

        for i in range(0, len(self.user_messages)):
            t = {"user": self.user_messages[i]}

            if i < len(self.system_messages):
                t["assistant"] = self.system_messages[i]
            output.append(t)
        return output

    def get_last_user_message(self):
        return self.user_messages[-1]

    def get_last_system_message(self):
        return self.system_messages[-1]


class InstructionFollower(Player):

    def __init__(self, model_name):
        super().__init__(model_name)

    def __call__(self, instruction: Instruction, turn_idx):
        return super().__call__(instruction.convert_to_query_messages(), turn_idx)

    def _custom_response(self, messages, turn_idx):
        return "Answer: first"


class InstructionGiver(Player):

    def __init__(self, model_name):
        super().__init__(model_name)

    def __call__(self, instruction: Instruction, turn_idx):
        return super().__call__(instruction.convert_to_query_messages(), turn_idx)

    def _custom_response(self, messages, turn_idx):
        return "Expression: Text"


class PolicyGeneratorGame:

    def __init__(self, game_instance: Dict, player_backends: List[str]):
        self.player_backends = player_backends
        self.game_id = game_instance['game_id']
        self.prompt = game_instance['prompt']
        self.instruction_giver = InstructionGiver(player_backends[0])

        self.given_instruction = Instruction()

        self.turn_count = 0


    def proceeds(self) -> bool:
        return True

    def turn(self):
        # generate referring expression - A side

        self.given_instruction.add_user_message(self.prompt)
        player_1_prompt, player_1_response, player_1_response_text = self.instruction_giver(self.given_instruction, None)
        # log the Player 1
        self.given_instruction.add_system_message(player_1_response_text)

        self.turn_count += 1