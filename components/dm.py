from components.utils import LLMGenerator, json_to_string
from components.dst import *
from components.prompts import SYSTEM_PROMPTS, USER_PROMPTS


class DialogueManager(LLMGenerator):
    """A class to generate the next best action."""

    def __init__(self, dialogue_state: DialogueState, generate_fn, args):
        super().__init__(dialogue_state, generate_fn, args)

    def get_clarification_data(self, action: str) -> dict:
        if self.args.debug: print(f"\n\033[95mDialogueManager.get_clarification_data: next action {action} is a request_clarification.\033[0m")
        invalid_slots_str = action.split(":")[1].strip() if ":" in action else ""
        if self.args.debug: print(f"\n\033[94mDialogueManager.get_clarification_data: invalid slots for clarification: {invalid_slots_str}\033[0m")
        return invalid_slots_str
    
    def get_slot_options(self, action: str, intent: str) -> dict:
        if self.args.debug: print(f"\n\033[95mDialogueManager.get_slot_options: next action {action} is a request_info. Searching for {intent}...\033[0m")
        intent_slots = SLOTS[intent] if intent in SLOTS else {}
        slot_options = {slot: value for slot, value in intent_slots.items() if action == f"request_info({slot})"}
        if not slot_options:
            if self.args.debug: print(f"\033[93mWarning: DialogueManager.get_slot_options: slots not found for {action}, using all intent slots.\033[0m")
            slot_options = intent_slots
        return slot_options
    
    def get_confirmation_data(self, action: str, intent: str, slots: dict) -> dict:
        if self.args.debug: print(f"\n\033[94mDialogueManager.get_confirmation_data: next action {action} is a confirmation, retrieving confirmation data.\033[0m")
        if intent in ["buy_tickets", "buy_merchandise"]:
            confirmation_data = get_price(intent, slots)
        elif intent in ["matches_schedule"]:
            confirmation_data = get_matches(team=slots.get("team", "male"))
        elif intent in ["get_news"]:
            confirmation_data = get_articles(team=slots.get("team", "male"), lang=self.args.user_settings["language"])
        elif intent in ["get_results"]:
            confirmation_data = get_results(team=slots.get("team", "male"))
        elif intent in ["information_request"]:
            topic = slots.get("topic")
            intent_requested = f"buy_{topic}"
            slot_requested = slots.get("specific_field")
            confirmation_data = self.get_slot_options(f"request_info({slot_requested})", intent_requested)
        return confirmation_data

    def check_frame_slots(self, frame_id: int) -> bool:
        """
        Check if all slots values in the frame are accepted and consistent.
        If a slot value is not accepted, it is set to None.
        If a slot value is inconsistent (e.g., date not valid for the team), it is set to None.
        Returns the list of invalid slots (with values).
        """
        invalid_slots = []
        frame = self.dialogue_state.get_state(frame_id)
        if frame is not None:
            intent = frame.get("intent")
            for slot, value in frame.get("slots", {}).items():
                if value is None or value == "" or value == "null":
                    continue
                if (intent == "buy_tickets" and slot == "date"):
                    if not is_valid_match_date(value, frame):
                        frame["slots"][slot] = None
                        invalid_slots.append((slot, value))
                elif not is_slot_value_accepted(intent, slot, value, frame):
                    frame["slots"][slot] = None
                    invalid_slots.append((slot, value))
        return invalid_slots

    def next_frame_actions(self, frame_id: int) -> dict:
        """Generate the next actions for a specific frame."""
        frame = self.dialogue_state.get_state(frame_id)

        # 1. Check if out_of_domain intent is detected
        if frame.get("intent") == "out_of_domain":
            if self.args.debug: print(f"\n\033[94mDialogueManager.next_frame_actions: out_of_domain intent detected in frame {frame_id}.\033[0m")
            return {"next_action": "fallback_policy"}

        # 2. Check frame slots, if invalid slots, next action is to ask for clarification
        if not self.args.evaluate: # In evaluation we can't know if slots are valid or not
            invalid_slots = self.check_frame_slots(frame_id)
            if invalid_slots:
                if self.args.debug: print(f"\n\033[94mDialogueManager.next_frame_actions: invalid slots {invalid_slots} in frame {frame_id}, asking for clarification.\033[0m")
                actions = {"next_action": f"request_clarification; invalid slots: {invalid_slots}"}
        else:
            invalid_slots = False  # In evaluation, we assume all slots are valid

        # 3. If all slots are valid, generate the next actions using LLM
        if not invalid_slots:
            system_prompt = SYSTEM_PROMPTS["DM_FEW_SHOTS"].format(list_of_slots=frame.get("slots", {}).keys())
            user_prompt = USER_PROMPTS["DM"].format(dialogue_state=json_to_string(frame))
            if self.args.debug: print(f"\n\033[94mDialogueManager.next_frame_actions: System Prompt: {system_prompt}\033[0m")
            if self.args.debug: print(f"\n\033[94mDialogueManager.next_frame_actions: User Prompt: {user_prompt}\033[0m")
            actions = self.generate_json(system_prompt, user_prompt)
            # Remove None actions
            actions = {k: v for k, v in actions.items() if v not in [None, {}, "None", "null"]}

        if self.args.debug: print(f"\n\033[94mDialogueManager.next_frame_actions: for {self.dialogue_state.get_state(frame_id)} -> {actions}\033[0m")
        return actions

    def execute(self, using_policy: bool = False) -> dict:
        """
        Choose the appropriate frame and generate the next best actions.
        If needed, retrieve information from external sources.
        
        """

        return_dict = {
            "frame_id": None,
            "next_actions": {},
            "actions_by_frame": {},
            "slot_options": {},
            "confirmation_data": {},
            "clarification_data": None
        }

        # 1. Compute next best actions for last frame
        last_frame_id = next(reversed(self.dialogue_state.frames), None)
        if self.args.debug: print(f"DialogueManager.execute: last frame ID is {last_frame_id}")
        # NOTE: Alternative, use the most recent intent of the user (i.e., largest frame ID)
        # last_frame_id = max(self.dialogue_state.get_dialogue_state().keys(), default=-1)

        actions = self.next_frame_actions(last_frame_id)
        return_dict["frame_id"] = last_frame_id
        return_dict["next_actions"] = actions
        return_dict["actions_by_frame"][last_frame_id] = actions

        intent = self.dialogue_state.get_state(last_frame_id).get("intent", None)
        slots = self.dialogue_state.get_state(last_frame_id).get("slots", {})

        # 2. For each action, retrieve slot options, confirmation data, and clarification data if needed
        if not self.args.evaluate:
            for _, action in actions.items():
                if self.args.debug: print(action)
                if action.startswith("fallback_policy"):
                    if self.args.debug: print(f"\n\033[94mDialogueManager.execute: next action {action} is a fallback policy, skip NLG.\033[0m")
                    continue
                if action.startswith("request_clarification"):
                    return_dict["clarification_data"] = self.get_clarification_data(action)
                if action.startswith("request_info"):
                    return_dict["slot_options"][action] = self.get_slot_options(action, intent)
                if action.startswith("confirmation"):
                    return_dict["confirmation_data"][action] = self.get_confirmation_data(action, intent, slots)

        return return_dict
