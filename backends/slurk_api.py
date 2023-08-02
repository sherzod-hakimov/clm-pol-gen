from typing import List, Dict, Tuple, Any

import requests
import socketio
from retry import retry

import backends

logger = backends.get_logger(__name__)

SUPPORTED_MODELS = ["slurk"]

NAME = "slurk"


class Slurk(backends.Backend):

    def __init__(self):
        creds = backends.load_credentials(NAME)
        assert "uri" in creds[NAME]
        if "uri" in creds[NAME]:
            self.uri = creds[NAME]["uri"]
        self.api_key = creds[NAME]["api_key"]  # slurk admin token
        self.clem_bot = None
        self.slurk_api = SlurkApi(self.api_key, self.uri)

    def prepare_and_wait_for_participant(self, task_room_layout, bot_permissions, user_permissions):
        """ This is called once for each game """

        layout_id = self.slurk_api.create_room_layout(task_room_layout)
        bot_permissions_id = self.slurk_api.create_permissions(bot_permissions)
        user_permissions_id = self.slurk_api.create_permissions(user_permissions)

        # We only need a single bot for now, because we only can serve a single room. See note below.
        # Could and should we make use of the waiting room? Then we would spawn a clembot for each room.
        room_id = self.slurk_api.create_room(layout_id)
        bot_token = self.slurk_api.create_token(bot_permissions_id, room_id)
        bot_id = self.slurk_api.create_user("clem_bot", bot_token)
        user_token = self.slurk_api.create_token(user_permissions_id, room_id)

        self.clem_bot = ClemBot(bot_id, bot_token, room_id).connect(self.uri)
        self.slurk_api.join_room(bot_id, room_id)  # todo why do we need this?

        logger.info(f"Use token to join clembench in slurk: {user_token}")
        self.clem_bot.wait_for_participant()
        logger.info("User joined")

    @retry(tries=3, delay=0, logger=logger)
    def generate_response(self, messages: List[Dict], model: str) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: chat-gpt for chat-completion, otherwise text completion
        :return: the continuation
        """
        response_text = self.clem_bot.wait_for_user_response(messages)
        return messages, {"response": "slurk"}, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS


class SlurkApi:

    def __init__(self, api_token: str, uri: str):
        self.uri = uri
        self.uri += "/slurk/api"
        self.authorize_header = {"Authorization": f"Bearer {api_token}"}

    def create_user(self, name, token):
        response = requests.post(
            f"{self.uri}/users", headers=self.authorize_header, json={"name": name, "token_id": token})
        self.assert_response(response, f"create user: {name}")
        return response.json()["id"]

    def create_task(self, name: str, num_users: int, layout_id: int):
        response = requests.post(f"{self.uri}/tasks", headers=self.authorize_header,
                                 json={"name": name, "num_users": num_users, "layout_id": layout_id})
        self.assert_response(response, f"create task: {name}")
        return response.json()["id"]

    def create_permissions(self, permissions: Dict):
        response = requests.post(f"{self.uri}/permissions", headers=self.authorize_header, json=permissions)
        self.assert_response(response, "create permissions")
        return response.json()["id"]

    def create_token(self, permissions_id: int, room_id: int, task_id: int = None):
        response = requests.post(
            f"{self.uri}/tokens", headers=self.authorize_header,
            json={
                "permissions_id": permissions_id,
                "room_id": room_id,
                "registrations_left": 1,
                "task_id": task_id,
            },
        )
        self.assert_response(response, "create token")
        return response.json()["id"]

    def create_room(self, room_layout_id: int):
        response = requests.post(
            f"{self.uri}/rooms", headers=self.authorize_header,
            json={"layout_id": room_layout_id},
        )
        self.assert_response(response, f"create room")
        return response.json()["id"]

    def create_room_layout(self, room_layout: Dict):
        response = requests.post(f"{self.uri}/layouts", headers=self.authorize_header, json=room_layout)
        self.assert_response(response, f"create room layout")
        return response.json()["id"]

    def join_room(self, user_id: int, room_id: int):
        response = requests.post(f"{self.uri}/users/{user_id}/rooms/{room_id}", headers=self.authorize_header)
        self.assert_response(response, f"user {user_id}  joins room {room_id}")

    def assert_response(self, response, description):
        if not response.ok:
            logger.error(f"`{description}` unsuccessful: {response.status_code}")
            response.raise_for_status()
        logger.debug(f"`{description}` successful.")


class ClemBot:

    def __init__(self, user_id: int, user_token: str, room_id: int):
        self.user_id = user_id
        self.user_token = user_token
        self.sio = socketio.Client(logger=True)
        self.sync_event = self.sio.eio.create_event()  # the vehicle to wait until user responds

        """
            TODO: For now we can only allow a single room and player.
            What would it mean to start the benchmark once, but let multiple users accomplish it?
            For this we would need a separate approach that starts an individual benchmark run
            when a user connects.
        """
        self.room_id = room_id
        self.user_messages = list()  # we need an object to carry over the response between threads

        def store_and_unblock(data):
            if data['room'] != self.room_id:
                return
            if data["user"]["id"] == self.user_id:
                return  # ignore self
            self.user_messages.append(data["message"])  # collect user response
            self.sync_event.set()  # continue the other thread

        self.sio.on("text_message", store_and_unblock)

        def check_and_unblock(data):
            if data['room'] != self.room_id:
                return
            if data["user"]["id"] == self.user_id:
                return  # ignore self
            self.sync_event.set()  # continue the other thread

        self.sio.on("status", check_and_unblock)

    def wait_for_user_response(self, messages) -> str:
        latest_response = "Nothing has been said yet."
        if messages:
            latest_response = messages[-1]["content"]
        self.sio.emit("text", {"message": latest_response, "room": self.room_id})
        if not self.sync_event.wait(timeout=5 * 60):  # seconds
            pass  # no user response
        self.sync_event.clear()
        user_response = self.user_messages[0]
        self.user_messages.clear()
        return user_response

    def wait_for_participant(self):
        if not self.sync_event.wait(timeout=5 * 60):  # seconds
            raise RuntimeError("no user joined the slurk room")
        self.sync_event.clear()

    def connect(self, uri):
        """Establish a connection to the remote server."""
        self.sio.connect(uri,
                         headers={"Authorization": f"Bearer {self.user_token}", "user": str(self.user_id)},
                         namespaces="/")
        return self
