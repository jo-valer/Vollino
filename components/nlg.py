from components.utils import LLMGenerator, json_to_string
from components.dst import DialogueState
from components.prompts import SYSTEM_PROMPTS, USER_PROMPTS


class NLG(LLMGenerator):
    """A class to generate natural language responses, based on the next actions chosen by the DM."""

    def __init__(self, dialogue_state: DialogueState, generate_fn, args):
        super().__init__(dialogue_state, generate_fn, args)

    def generate_response(self, dm_json: dict):
        """Generate a response based on the next actions."""
        frame_id = dm_json.get("frame_id")
        intent = self.dialogue_state.get_state(frame_id).get("intent", None)
        if intent is None:
            raise ValueError(f"No intent found for frame ID {frame_id}")
        next_actions = dm_json.get("next_actions")

        clarification_data = dm_json.get("clarification_data")
        next_actions_options = dm_json.get("slot_options")
        confirmation_data = dm_json.get("confirmation_data")
        next_actions = dm_json.get("next_actions")

        if clarification_data: # Use NLG_CLARIFICATION prompt for request_clarification actions
            if self.args.debug: print(f"Using NLG_CLARIFICATION prompt for next actions: {json_to_string(clarification_data)}")
            user_prompt = USER_PROMPTS["NLG_CLARIFICATION"].format(
                dialogue_state=json_to_string(self.dialogue_state.get_state(frame_id)),
                next_actions=json_to_string(next_actions),
                invalid_slots=json_to_string(clarification_data)
            )
        elif next_actions_options: # Use NLG_REQUEST_INFO prompt for request_info actions
            options = json_to_string(next_actions_options, no_brackets=True)
            if self.args.debug: print(f"Using NLG_REQUEST_INFO prompt for next actions: {options}")
            user_prompt = USER_PROMPTS["NLG_REQUEST_INFO"].format(
                dialogue_state=json_to_string(self.dialogue_state.get_state(frame_id)),
                next_actions=json_to_string(next_actions),
                options=options
            )
        elif confirmation_data: # Use NLG_CONFIRMATION prompt for confirmation actions
            if self.args.debug: print(f"Using NLG_CONFIRMATION prompt for confirmation data: {json_to_string(confirmation_data)}")
            user_prompt = USER_PROMPTS["NLG_CONFIRMATION"].format(
                dialogue_state=json_to_string(self.dialogue_state.get_state(frame_id)),
                next_actions=json_to_string(next_actions),
                confirmation_data=json_to_string(confirmation_data)
            )
        else: # No request_info actions, nor confirmation actions
            if self.args.debug: print(f"\033[93mWarning: NLG.generate_response: no request_info or confirmation actions found for frame ID {frame_id}, using default user prompt.\033[0m")
            next_actions_options = next_actions.get(frame_id, {})
            user_prompt = USER_PROMPTS["NLG"].format(
                dialogue_state=json_to_string(self.dialogue_state.get_state(frame_id)),
                next_actions=json_to_string(next_actions),
            )
        if self.args.debug: print(f"NLG.generate_response: User prompt:\n{user_prompt}")

        system_prompt = SYSTEM_PROMPTS["NLG"].format(intent=intent)
        response = self.generate_json(system_prompt, user_prompt)
        if self.args.debug: print(f"NLG.generate_response: {response}")
        if response is None:
            if self.args.debug: print(f"\033[93mWarning: NLG.generate_response: no response found, giving up...\033[0m")
            return None

        return response.get("response", None)
    
    def merge_responses(self, responses: list):
        """Merge multiple responses into a single coherent response."""
        # TODO: Verify if merging with an LLM is necessary/the best idea
        pass
