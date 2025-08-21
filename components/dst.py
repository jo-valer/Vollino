import json

from data.data import *


class DialogueState:
    """A class to represent the dialogue state of the conversational agent.
    It consists of a list of frames, each representing an intention.
    Each frame is a dictionary and is identified by a unique key."""

    def __init__(self, args):
        self.args = args
        self.frames = {}
        self.dialogue_history = {
            "conversation_history": [],
            "action_history": []
        }
        self.conversation_history = self.dialogue_history["conversation_history"]
        self.action_history = self.dialogue_history["action_history"]
        self.last_confirmed_frame = None
    
    def __str__(self):
        return json.dumps({
            "frames": self.frames,
            "conversation_history": self.conversation_history,
            "action_history": self.action_history,
            "last_confirmed_frame": self.last_confirmed_frame
        }, indent=2)

    def __repr__(self):
        return self.__str__()
    
    def is_empty(self):
        """Check if the dialogue state is empty."""
        return len(self.frames) == 0

    def clean_conversation_history(self):
        """Remove all entries from the conversation history."""
        self.conversation_history.clear()

    def accept_slot(self, intent: str, slot: str, value: str):
        """Check if the slot can be accepted for the given intent, i.e., if:
        - the slot is among the possible slots of the intent
        - the value is not null (NB. False is valid)"""
        if intent in SLOTS:
            if slot in SLOTS[intent]:
                if value is not None:
                    return True
        return False

    def prefilling_slot_value(self, frame: dict, slot: str):
        """
        Get prefill value for the slot with user settings info (if available). This works for:
        - delivery_address
        - any slot with same name in last confirmed frame

        Otherwise, it returns None as prefill value.
        """
        if slot == "delivery_address":
            buy_history = self.args.user_settings.get("buy_history")
            if buy_history:
                last_purchase = next((purchase for purchase in reversed(buy_history) if purchase.get("intent") == "buy_merchandise"), None)
                if last_purchase:
                    address = last_purchase.get("slots", {}).get("delivery_address")
                    if self.args.debug: print(f"\033[92mInfo: DialogueState.prefill_slot: using delivery address from user settings.\033[0m")
                    return address
        elif frame.get("intent") == "buy_merchandise":
            item = frame.get("slots", {}).get("item", "")
            if item and item.lower() != "shirt" and slot == "size":
                if self.args.debug: print(f"\033[92mInfo: DialogueState.prefill_slot: defaulting size to 'unique' because item is not a shirt.\033[0m")
                return DEFAULT_SIZE
        
        # Check if the slot is in the last confirmed frame
        if self.last_confirmed_frame and slot in self.last_confirmed_frame.get("slots", {}):
            value = self.last_confirmed_frame["slots"][slot]
            if value is not None and value != "":
                if self.args.debug: print(f"\033[92mInfo: DialogueState.prefill_slot: using last confirmed frame value for slot '{slot}': {value}\033[0m")
                return value
        return None

    def filter_frame(self, frame: dict):
        """Filter the frame to only include slots accepted for the given intent."""
        intent = frame.get("intent")
        if intent in SLOTS:
            accepted_slots = {slot: value for slot, value in frame.get("slots", {}).items()
                              if self.accept_slot(intent, slot, value)}
            frame["slots"] = accepted_slots
        return frame

    def move_frame_to_end(self, frame_id):
        """Move a frame with the given frame_id to the end of the frames dictionary."""
        if frame_id in self.frames:
            # Remove the frame and reinsert it to preserve order
            frame = self.frames.pop(frame_id)
            self.frames[frame_id] = frame
        else:
            raise KeyError(f"Frame ID '{frame_id}' not found in frames.")

    def reorder_frames(self):
        """
        Reorder the frames based on the developer-defined order of intents.
        The last frame should be the most relevant.
        Note: frames keep their id when moved.
        """
        # Just move information_request frames to the end proved to be ineffective
        # NOTE: A better alternative is to leave frames in the order of creation (the idea is that the last one is the most relevant)
        # for frame_id in list(self.frames.keys()):
        #     if self.frames[frame_id].get("intent") == "information_request":
        #         self.move_frame_to_end(frame_id)
        pass

    def add_state(self, frame: dict):
        """Create a new unique key and add the frame to the dialogue state."""
        # key = len(self.frames)
        biggest_key = max(self.frames.keys(), default=-1)
        key = biggest_key + 1
        if frame.get("intent") != "out_of_domain": # If not out_of_domain, filter and add missing slots
            frame = self.filter_frame(frame) # NOTE: This also removes slots with value None
            # Add missing slots
            for slot in SLOTS.get(frame.get("intent"), []):
                if slot not in frame.get("slots", {}):
                    frame["slots"][slot] = self.prefilling_slot_value(frame, slot)
                elif isinstance(frame["slots"][slot], str) and slot != "delivery_address": # Lowercase all existing string slots except delivery_address
                    frame["slots"][slot] = frame["slots"][slot].lower()
        self.frames[key] = frame

    def remove_state(self, key: int):
        """Remove a frame from the dialogue state."""
        if key in self.frames:
            del self.frames[key]
    
    def set_last_confirmed_frame(self, frame: dict):
        """Set the last confirmed frame."""
        self.last_confirmed_frame = frame.copy() if frame else None

    def merge_state(self, frame: dict):
        """Merge the frame into the dialogue state.
        If the intent already exists, it will be updated and moved to the end."""
        # If out_of_domain intent is detected, add it without checking
        if frame.get("intent") == "out_of_domain":
            self.add_state(frame)
            return
        matched = False
        matched_frame_id = None
        for frame_id, loc_frame in self.frames.items():
            if loc_frame.get("intent") == frame.get("intent"):
                for slot, value in frame.get("slots", {}).items():
                    if self.accept_slot(loc_frame.get("intent"), slot, value):
                        if isinstance(value, str) and slot != "delivery_address":
                            value = value.lower()
                        loc_frame["slots"][slot] = value
                matched = True
                matched_frame_id = frame_id
                break
        if matched: # Move the updated frame to the end
            self.move_frame_to_end(matched_frame_id)
        else:
            self.add_state(frame)

    def get_state(self, key: int):
        """Get a frame by its key."""
        return self.frames.get(key)
    
    def get_dialogue_state(self) -> dict:
        """Get the entire dialogue state (dictionary of frames)."""
        return self.frames

    def get_state_list(self) -> list:
        """Return the list of frames."""
        return list(self.frames.values())

    def add_turn(self, text: str, turn_type: str):
        """Add a turn to the conversation history."""
        self.conversation_history.append({"type": turn_type, "text": text})
    
    def get_conversation_history(self, as_string: bool = False):
        """Get the conversation history."""
        if as_string:
            return "\n".join(f"- {turn['type']}: {turn['text']}" for turn in self.conversation_history)
        return self.conversation_history

    def reorder_actions(self):
        """Reorder the actions based on the developer-defined order of actions."""
        # TODO: Implement this method, note that the order list entries do not match the action intents (as these are generic, not instantiated)
        pass

    def add_actions(self, actions: dict):
        """Add the dictionary of next actions to the action history."""
        for frame_id, action in actions.items():
            self.action_history.append({"frame_id": frame_id, "actions": action})
    
    def get_actions(self, as_string: bool = False):
        """Get the action history."""
        if as_string:
            return "\n".join(f"- {action}" for action in self.action_history)
        return self.action_history
        

class DialogueStateTracker:
    """A class to track the dialogue state of the conversational agent.
    It provides methods to add, merge, and retrieve the dialogue state."""

    def __init__(self, dialogue_state: DialogueState, args):
        self.dialogue_state = dialogue_state
        self.args = args

    def update_state(self, frames: list):
        """Update the dialogue state with a list of frames, and reorder the frames."""
        for frame in frames:
            self.dialogue_state.merge_state(frame)
        # self.dialogue_state.reorder_frames() # NOTE: For now, best experience by keeping the most recent frame as most relevant
        if self.args.debug: print(f"\n\033[95mDialogueStateTracker.update_state: {self.dialogue_state.get_dialogue_state()}\033[0m")

    def update_actions(self, actions: dict):
        """Update the dialogue state with a dictionary of next actions, and reorder the actions."""
        self.dialogue_state.add_actions(actions)
        self.dialogue_state.reorder_actions()
        if self.args.debug: print(f"\n\033[95mDialogueStateTracker.update_actions: {self.dialogue_state.get_actions(as_string=True)}\033[0m")
    
    def clear_state_after_turn(self):
        """Clear the dialogue state after a turn. Removing frames and conversation history when needed."""
        
        # If one of the next actions has "confirmation" in its value, remove the corresponding frame from dialogue state after saving the purchase in the user_settings
        action_history = self.dialogue_state.get_actions()
        last_actions = action_history[-1] if action_history else None
        for action in last_actions.get("actions", {}).values() if last_actions else []:
            if "confirmation" in action:
                frame_id = last_actions.get("frame_id")
                # Save the frame to user settings (if the intent is buy_<item>)
                intent = self.dialogue_state.get_state(frame_id).get("intent")
                if "buy_" in intent:
                    buy_history = self.args.user_settings.get("buy_history", [])
                    buy_history.append({
                        "intent": intent,
                        "slots": self.dialogue_state.get_state(frame_id).get("slots", {})
                    })
                    self.args.user_settings["buy_history"] = buy_history
                    # Update the file
                    update_user_settings(self.args.user_settings)

                # Move the frame from dialogue state to confirmed state
                self.dialogue_state.set_last_confirmed_frame(self.dialogue_state.get_state(frame_id))
                self.dialogue_state.remove_state(frame_id)
                if self.args.debug: print(f"\n\033[95mDialogueStateTracker.clear_state_after_turn: moved frame {frame_id} to confirmed state.\033[0m")

                # Clean the conversation history
                self.dialogue_state.clean_conversation_history()

        # If one of the next actions has "fallback_policy" in its value, remove the corresponding frame from dialogue state
        for action in last_actions.get("actions", {}).values() if last_actions else []:
            if "fallback_policy" in action:
                frame_id = last_actions.get("frame_id")
                self.dialogue_state.remove_state(frame_id)
                if self.args.debug: print(f"\n\033[95mDialogueStateTracker.clear_state_after_turn: removed frame {frame_id} due to fallback_policy action.\033[0m")
