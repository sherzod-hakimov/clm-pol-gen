import time
import unittest

from backends import openai_api
from clemgame.slurkbot import APIChatBot

from clemgame import benchmark, string_utils


class MyTestCase(unittest.TestCase):
    """
    on user joins: wait for start benchmark command

    game description is prompted + possible history
    waiting for player response
    """

    def test_run_taboo(self):
        benchmark.run(game_name="taboo", temperature=0.0,
                      experiment_name="low_en",
                      model_name=string_utils.to_pair_descriptor([openai_api.MODEL_GPT_35, "slurk"]))

    def test_something(self):
        slurk_host = "http://0.0.0.0"
        slurk_port = "5000"
        slurk_token = "<here>"
        slurk_task = None
        slurk_user = 1
        timeout = 60

        bot = APIChatBot(slurk_token, slurk_user, slurk_task, slurk_host, slurk_port)
        bot.run()  # we connect but do not wait

        callback_event = bot.sio.eio.create_event()
        user_response = []  # we need an object to carry over the response between threads

        def message(data):
            user_id = data["user"]["id"]
            if user_id == 1:
                return
            user_response.append(data["message"])
            callback_event.set()

        bot.sio.on("text_message", message)

        for turn in range(5):
            bot.sio.emit("text", {
                "message": f"prompt: turn {turn}",
                "room": 1
            })
            if not callback_event.wait(timeout=timeout):
                pass  # no user response
            callback_event.clear()
            print(user_response[0])
            user_response.clear()


if __name__ == '__main__':
    unittest.main()
