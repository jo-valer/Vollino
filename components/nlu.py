from components.utils import LLMGenerator, extract_json, json_to_string
from components.dst import DialogueState, SLOTS, is_valid_specific_field
from components.prompts import SYSTEM_PROMPTS, USER_PROMPTS
    

class NLU(LLMGenerator):
    """A class to extract slot values from user input based on the intent."""

    def __init__(self, dialogue_state: DialogueState, generate_fn, args):
        super().__init__(dialogue_state, generate_fn, args)

    def _chunk(self, user_input: str):
        """Chunk the user input into different parts based on the dialogue state."""
        if self.dialogue_state.is_empty():
            system_prompt = SYSTEM_PROMPTS["CHUNKER"]
        else: 
            intent = self.dialogue_state.get_state_list()[0].get("intent", "unknown")
            system_prompt = SYSTEM_PROMPTS["CHUNKER_INTENT_CONDITIONED"].format(intent=intent)
        conversation_history = self.dialogue_state.get_conversation_history(as_string=True)
        user_prompt = USER_PROMPTS["CHUNKER"].format(conversation_history=conversation_history, user_input=user_input)
        response = self.generate_json(system_prompt, user_prompt)
        if self.args.debug: print(f"\n\033[94mNLU._chunk: {response}\033[0m")
        return response

    def _slots_from_chunk(self, intent: str, chunk: str):
        """Extract the intent and slots from a single chunk of the user input."""
        slots = SLOTS[intent] if intent in SLOTS else {}
        if intent == "information_request":
            valid_specific_fields_list = is_valid_specific_field(None, None, return_list=True)
            system_prompt = SYSTEM_PROMPTS["NLU_SPECIFIC_FIELD"].format(intent=intent, specific_fields=str(valid_specific_fields_list))
        else:                
            system_prompt = SYSTEM_PROMPTS["NLU"].format(intent=intent, slots=json_to_string(slots))
        user_prompt = USER_PROMPTS["NLU"].format(
            conversation_history=self.dialogue_state.get_conversation_history(as_string=True),
            user_input=chunk)
        # if self.args.debug: print(f"\n\033[94mNLU._slots_from_chunk: System Prompt: {system_prompt}\033[0m")
        # if self.args.debug: print(f"\n\033[94mNLU._slots_from_chunk: User Prompt: {user_prompt}\033[0m")
        output = self.generate_json(system_prompt, user_prompt)
        if self.args.debug: print(f"\n\033[94mNLU._slots_from_chunk: {output}\033[0m")
        return output

    def intents_and_slots(self, user_input: str) -> list[dict]:
        """Chunk the user input and extract intents and slots for each chunk."""
        chunks = self._chunk(user_input)
        intents_and_slots = []
        for intent, chunk in chunks.items():
            if intent == "out_of_domain":
                slots = {"user_input": chunk}
            else:
                slots = self._slots_from_chunk(intent, chunk)
            intents_and_slots.append({"intent": intent, "slots": slots})
        if self.args.debug: print(f"\n\033[94mNLU.intents_and_slots: {intents_and_slots}\033[0m")
        return intents_and_slots

    def run_chunker(self, user_input: str) -> dict:
        """Run the chunker on the user input and return the chunks."""
        return self._chunk(user_input)
