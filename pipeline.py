import argparse
import torch

from argparse import Namespace
from functools import partial

from components.utils import load_model, set_client, generate, init_log_files, log_turn
from components.utils import MODELS, TEMPLATES
from components.prompts import DEFAULT_SYSTEM_TURNS
from components.dst import DialogueState, DialogueStateTracker
from data.data import get_user_settings
from components.dm import DialogueManager
from components.nlu import NLU
from components.nlg import NLG


def get_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m query_model",
        description="Query a specific model with a given input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model_name",
        type=str,
        choices=list(MODELS.keys()),
        help="The model to query.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to use for the model.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Split the model across multiple devices.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["f32", "bf16"],
        default="bf16",
        help="The data type to use for the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="The maximum sequence length to use for the model.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=0,
        help="The user ID to use for the conversation."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the model in evaluation mode."
    )
    parser.add_argument(
        "--eval-nlu-only",
        action="store_true",
        help="Run NLU evaluation only."
    )
    parser.add_argument(
        "--eval-dm-only",
        action="store_true",
        help="Run DM evaluation only."
    )
    parser.add_argument(
        "--eval-chunker-only",
        action="store_true",
        help="Run Chunker evaluation only."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode, i.e., print debug information and save logs."
    )
    parser.add_argument(
        "--avoid-input-translation",
        action="store_true",
        help="Avoid translating user input."
    )
    parser.add_argument(
        "--interface",
        action="store_true",
        help="Enable GUI interface instead of terminal input."
    )

    parsed_args = parser.parse_args()
    parsed_args.chat_template = TEMPLATES[parsed_args.model_name] if parsed_args.model_name in TEMPLATES else None
    parsed_args.model_id = MODELS[parsed_args.model_name]

    return parsed_args


def main():
    args = get_args()
    user_settings = get_user_settings(args.user_id)

    # Add user_settings to args
    args.user_settings = user_settings

    # Initialize the translator if the user speaks Italian
    if user_settings["language"] == "it":
        if args.debug: print("Initializing translator for Italian language...")
        from components.translator import Translator
        translator = Translator()
        if args.debug: print("Translator initialized.")
    else:
        if args.debug: print("No translator needed.")
        translator = None

    if args.model_name.startswith("groq"):
        client = set_client()
        generation_args = {
            "model_type": "groq",
            "client": client,
            "model": args.model_id,
            "args": args
        }
    else:
        model, tokenizer = load_model(args)
        generation_args = {
            "model_type": "local",
            "model": model,
            "tokenizer": tokenizer,
            "args": args
        }

    generate_fn = partial(generate, generation_args=generation_args)

    if args.evaluate:
        from evaluation.evaluate import evaluate_nlu_from_pipeline, evaluate_dm_from_pipeline, evaluate_chunker_from_pipeline
        
        # Initialize the components for evaluation
        dialogue_state = DialogueState(args)
        nlu = NLU(dialogue_state, generate_fn, args)
        dm = DialogueManager(dialogue_state, generate_fn, args)
        
        # Run NLU evaluation
        if not args.eval_dm_only and not args.eval_chunker_only:
            print("Starting NLU evaluation...")
            nlu_results = evaluate_nlu_from_pipeline(nlu, args, translator)

        # Run DM evaluation
        if not args.eval_nlu_only and not args.eval_chunker_only:
            print("\nStarting DM evaluation...")
            dm_results = evaluate_dm_from_pipeline(dm, args)

        # Run Chunker evaluation
        if not args.eval_nlu_only and not args.eval_dm_only:
            print("\nStarting Chunker evaluation...")
            chunker_results = evaluate_chunker_from_pipeline(nlu, args, translator)

        print("Evaluation completed successfully!")
        return
    
    else:
        # Initialize the components and dialogue state
        dialogue_state = DialogueState(args)
        dialogue_state_tracker = DialogueStateTracker(dialogue_state, args)
        nlu = NLU(dialogue_state, generate_fn, args)
        dm = DialogueManager(dialogue_state, generate_fn, args)
        nlg = NLG(dialogue_state, generate_fn, args)
        
        # Initialize log files
        if args.debug:
            print("\033[93mDebug mode enabled\033[0m")
            filenames, files_text = init_log_files()
        
        if args.interface:
            run_gui_interface(args, user_settings, dialogue_state, dialogue_state_tracker, 
                            nlu, dm, nlg, filenames if args.debug else None, 
                            files_text if args.debug else None)
        else:
            run_terminal_interface(args, user_settings, dialogue_state, dialogue_state_tracker, 
                                 nlu, dm, nlg, filenames if args.debug else None, 
                                 files_text if args.debug else None)


def run_terminal_interface(args, user_settings, dialogue_state, dialogue_state_tracker, 
                         nlu, dm, nlg, filenames=None, files_text=None):
    """Run the conversation in terminal mode."""
    # Initialize translator if needed
    if user_settings["language"] == "it":
        from components.translator import Translator
        translator = Translator()
    else:
        translator = None
    
    # Start the conversation loop
    if user_settings["new_user"]:
        last_system_turn = DEFAULT_SYSTEM_TURNS["INIT"]
    else:
        last_system_turn = DEFAULT_SYSTEM_TURNS["START"]

    turn = 0
    while True:
        # Print the last system turn
        if user_settings["language"] == "it" and translator: # Translate response en -> it only for UI
            last_system_turn_it = translator(last_system_turn, target_lang="it")
            print(f"\033[96mVollino\033[0m > \033[96m{last_system_turn_it}\033[0m")
        else:
            print(f"\033[96mVollino\033[0m > \033[96m{last_system_turn}\033[0m")

        # Wait for the user input
        user_input = input("\033[92mUser\033[0m > \033[92m")
        print("\033[0m") # Reset color after user input

        if user_settings["language"] == "it" and not args.avoid_input_translation and translator: # Translate user input it -> en for the agent
            user_input = translator(user_input, target_lang="en")

        # Process the turn
        last_system_turn = process_turn(user_input, last_system_turn, dialogue_state, 
                                       dialogue_state_tracker, nlu, dm, nlg, 
                                       args, turn, filenames, files_text)
        turn += 1


def run_gui_interface(args, user_settings, dialogue_state, dialogue_state_tracker, 
                     nlu, dm, nlg, filenames=None, files_text=None):
    """Run the conversation with GUI interface."""
    from gui_interface import VollinoGUI
    
    # Initialize translator if needed
    if user_settings["language"] == "it":
        from components.translator import Translator
        translator = Translator()
    else:
        translator = None
    
    # Initialize conversation state
    if user_settings["new_user"]:
        last_system_turn = DEFAULT_SYSTEM_TURNS["INIT"]
    else:
        last_system_turn = DEFAULT_SYSTEM_TURNS["START"]
    
    turn = 0
    
    def process_user_message(user_input):
        """Process user message and return system response."""
        nonlocal last_system_turn, turn
        
        # Translate user input if needed
        processed_input = user_input
        if user_settings["language"] == "it" and not args.avoid_input_translation and translator:
            processed_input = translator(user_input, target_lang="en")
        
        # Process the turn
        result = process_turn(processed_input, last_system_turn, dialogue_state, 
                             dialogue_state_tracker, nlu, dm, nlg, 
                             args, turn, filenames, files_text)
        
        last_system_turn = result["response"]
        turn += 1
        return result
    
    # Create and run GUI
    gui = VollinoGUI(
        callback_func=process_user_message,
        translator=translator,
        user_settings=user_settings
    )
    gui.add_initial_message(last_system_turn)
    gui.run()


def process_turn(user_input, last_system_turn, dialogue_state, dialogue_state_tracker, 
                nlu, dm, nlg, args, turn, filenames=None, files_text=None):
    """Process a single conversation turn."""
    # Update the conversation history
    dialogue_state.add_turn(text=last_system_turn, turn_type="system")
    dialogue_state.add_turn(text=user_input, turn_type="user")

    if args.debug: 
        print(f"Dialogue State: {dialogue_state.get_conversation_history(as_string=True)}")

    try:
        # NLU
        nlu_output = nlu.intents_and_slots(user_input)
        dialogue_state_tracker.update_state(nlu_output)

        # DM
        dm_output = dm.execute()
        dialogue_state_tracker.update_actions(dm_output.get("actions_by_frame", {}))

        # NLG
        system_response = nlg.generate_response(dm_output)

        # Check if there's a buy confirmation action for GUI checkout button
        show_checkout = False
        if args.interface:
            actions_by_frame = dm_output.get("actions_by_frame", {})
            for frame_id, actions in actions_by_frame.items():
                for action in actions.values():
                    if "confirmation" in action:
                        frame = dialogue_state.get_state(frame_id)
                        if frame and frame.get("intent") in ["buy_tickets", "buy_merchandise"]:
                            show_checkout = True
                            break
                if show_checkout:
                    break

        # Clear the dialogue state after the turn
        dialogue_state_tracker.clear_state_after_turn()

        # Log the turn if debug mode is enabled
        if args.debug and filenames and files_text: 
            log_turn(filenames, files_text, turn, user_input, nlu_output, dm_output, system_response, dialogue_state)

    except Exception as e:
        system_response = DEFAULT_SYSTEM_TURNS["FALLBACK"]
        show_checkout = False
        if args.debug: print(f"\033[31mError processing turn {turn}: {e}\033[0m")

    # Return response with checkout info for GUI
    if args.interface:
        return {"response": system_response, "show_checkout": show_checkout}
    else:
        return system_response


if __name__ == "__main__":
    main()
