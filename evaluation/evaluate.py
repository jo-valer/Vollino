import json
import os
import random
from sklearn.metrics import precision_recall_fscore_support

from components.dst import DialogueState
from components.prompts import DEFAULT_SYSTEM_TURNS


def values_match(pred_value, gt_value):
    """
    Check if predicted and ground truth values match, handling type conversions.
    """
    if pred_value is None and gt_value is None:
        return True
    if pred_value is None or gt_value is None:
        return False
    
    # Direct equality check first
    if pred_value == gt_value:
        return True
    
    # Convert both to strings and compare (handles int vs string cases)
    try:
        if str(pred_value).strip() == str(gt_value).strip():
            return True
    except:
        pass
    
    # Try numeric comparison if both can be converted to numbers
    try:
        pred_num = float(pred_value)
        gt_num = float(gt_value)
        return pred_num == gt_num
    except (ValueError, TypeError):
        pass
    
    # Case-insensitive string comparison
    try:
        if str(pred_value).strip().lower() == str(gt_value).strip().lower():
            return True
    except:
        pass
    
    return False


def load_nlu_testset(filepath):
    """Load the NLU test dataset from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_intent_classification(predictions, ground_truths):
    """
    Evaluate intent classification using precision, recall, and F1-score.
    Handles multiple frames (intents) per sentence.
    """
    total_correct = 0
    total_predicted = 0
    total_ground_truth = 0
    total_exact_matches = 0
    
    for pred_intents, gt_intents in zip(predictions, ground_truths):
        # Convert to sets for easier comparison
        pred_set = set(pred_intents) if isinstance(pred_intents, list) else {pred_intents}
        gt_set = set(gt_intents) if isinstance(gt_intents, list) else {gt_intents}
        
        # Count true positives (correctly predicted intents)
        correct_intents = pred_set.intersection(gt_set)
        total_correct += len(correct_intents)
        
        # Count total predictions and ground truths
        total_predicted += len(pred_set)
        total_ground_truth += len(gt_set)
        
        # Count exact matches (all intents correct)
        if pred_set == gt_set:
            total_exact_matches += 1
    
    # Calculate metrics
    precision = total_correct / total_predicted if total_predicted > 0 else 0.0
    recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = total_exact_matches / len(predictions) if len(predictions) > 0 else 0.0
    
    return {
        'intent_accuracy': accuracy,
        'intent_precision': precision,
        'intent_recall': recall,
        'intent_f1': f1,
        'total_correct': total_correct,
        'total_predicted': total_predicted,
        'total_ground_truth': total_ground_truth
    }


def evaluate_slot_filling(predictions, ground_truths, predicted_intents, ground_truth_intents):
    """
    Evaluate slot filling using precision, recall, and F1-score.
    Only evaluates slots for correctly predicted intents.
    """
    # Collect all unique slot names
    all_slots = set()
    valid_predictions = []
    valid_ground_truths = []
    
    for pred_frames, gt_frames, pred_intents, gt_intents in zip(predictions, ground_truths, predicted_intents, ground_truth_intents):
        # Convert to lists if single values
        if not isinstance(pred_frames, list):
            pred_frames = [pred_frames]
        if not isinstance(gt_frames, list):
            gt_frames = [gt_frames]
        if not isinstance(pred_intents, list):
            pred_intents = [pred_intents]
        if not isinstance(gt_intents, list):
            gt_intents = [gt_intents]
        
        # Create mapping from intent to slots for both predictions and ground truth
        pred_intent_to_slots = {}
        gt_intent_to_slots = {}
        
        for i, intent in enumerate(pred_intents):
            if i < len(pred_frames):
                slots = pred_frames[i]
                # Handle None values by replacing with empty dict
                pred_intent_to_slots[intent] = slots if slots is not None else {}
        
        for i, intent in enumerate(gt_intents):
            if i < len(gt_frames):
                slots = gt_frames[i]
                # Handle None values by replacing with empty dict
                gt_intent_to_slots[intent] = slots if slots is not None else {}
        
        # Only evaluate slots for correctly predicted intents
        correct_intents = set(pred_intents).intersection(set(gt_intents))
        
        for intent in correct_intents:
            pred_slots = pred_intent_to_slots.get(intent, {})
            gt_slots = gt_intent_to_slots.get(intent, {})
            
            # Handle None values by replacing with empty dict
            if pred_slots is None:
                pred_slots = {}
            if gt_slots is None:
                gt_slots = {}
            
            valid_predictions.append(pred_slots)
            valid_ground_truths.append(gt_slots)
            
            # Collect all slot names
            all_slots.update(pred_slots.keys())
            all_slots.update(gt_slots.keys())
    
    # If no valid predictions, return zero metrics
    if not valid_predictions:
        return {
            'slot_micro_precision': 0.0,
            'slot_micro_recall': 0.0,
            'slot_micro_f1': 0.0,
            'slot_macro_precision': 0.0,
            'slot_macro_recall': 0.0,
            'slot_macro_f1': 0.0,
            'per_slot_metrics': {},
            'valid_slot_evaluations': 0
        }
    
    # Calculate metrics for each slot
    slot_metrics = {}
    y_true_all = []
    y_pred_all = []
    
    for slot in all_slots:
        y_true_slot = []
        y_pred_slot = []
        
        for pred_slots, gt_slots in zip(valid_predictions, valid_ground_truths):
            # Handle None values by replacing with empty dict
            if pred_slots is None:
                pred_slots = {}
            if gt_slots is None:
                gt_slots = {}
                
            # Get predicted and ground truth values for this slot
            pred_value = pred_slots.get(slot, None)
            gt_value = gt_slots.get(slot, None)
            
            # Create binary classification: correct prediction vs incorrect/missing
            if gt_value is not None:
                y_true_slot.append(1)  # Slot should be present
                if values_match(pred_value, gt_value):
                    y_pred_slot.append(1)  # Correct prediction
                else:
                    y_pred_slot.append(0)  # Incorrect or missing prediction
            else:
                if pred_value is not None:
                    # False positive: predicted a slot that shouldn't be there
                    y_true_slot.append(0)
                    y_pred_slot.append(1)
        
        # Add to overall lists for micro averaging
        y_true_all.extend(y_true_slot)
        y_pred_all.extend(y_pred_slot)
        
        # Calculate metrics for this slot if we have data
        if y_true_slot:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_slot, y_pred_slot, average='binary', zero_division=0
            )
            slot_metrics[slot] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    # Calculate micro averages (overall performance)
    if y_true_all:
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true_all, y_pred_all, average='binary', zero_division=0
        )
    else:
        micro_precision = micro_recall = micro_f1 = 0.0
    
    # Calculate macro averages (average across slots)
    if slot_metrics:
        macro_precision = sum(metrics['precision'] for metrics in slot_metrics.values()) / len(slot_metrics)
        macro_recall = sum(metrics['recall'] for metrics in slot_metrics.values()) / len(slot_metrics)
        macro_f1 = sum(metrics['f1'] for metrics in slot_metrics.values()) / len(slot_metrics)
    else:
        macro_precision = macro_recall = macro_f1 = 0.0
    
    return {
        'slot_micro_precision': micro_precision,
        'slot_micro_recall': micro_recall,
        'slot_micro_f1': micro_f1,
        'slot_macro_precision': macro_precision,
        'slot_macro_recall': macro_recall,
        'slot_macro_f1': macro_f1,
        'per_slot_metrics': slot_metrics,
        'valid_slot_evaluations': len(valid_predictions)
    }


def evaluate_nlu(nlu_model, test_data, args, translator=None):
    """
    Evaluate the NLU component on the test dataset.
    Handles multiple frames (intents) per sentence.
    """
    print(f"Evaluating NLU on {len(test_data)} examples...")
    
    predicted_intents_list = []
    ground_truth_intents_list = []
    predicted_slots_list = []
    ground_truth_slots_list = []
    predictions_data = []  # Store detailed predictions for saving to file
    
    for i, example in enumerate(test_data):
        if i % 50 == 0:
            print(f"Processing example {i}/{len(test_data)}...")
        
        if translator:
            user_input = example['user_input_it']
            if not args.avoid_input_translation:
                user_input = translator(user_input, target_lang="en")
        else:
            user_input = example['user_input']
        ground_truth = example['ground_truth']
        
        # Get NLU predictions
        try:
            # Reset dialogue state for each example
            dialogue_state = DialogueState(args)
            # Add initial system turn to conversation history
            if translator:
                dialogue_state.add_turn(translator(DEFAULT_SYSTEM_TURNS["INIT"], target_lang="en"), "system")
            else:
                dialogue_state.add_turn(DEFAULT_SYSTEM_TURNS["INIT"], "system")
            nlu_model.dialogue_state = dialogue_state
            
            nlu_outputs = nlu_model.intents_and_slots(user_input)
            
            # Extract intents and slots from all outputs (handle multiple frames)
            predicted_intents = []
            predicted_slots = []
            
            if nlu_outputs:
                for output in nlu_outputs:
                    predicted_intents.append(output.get('intent', 'unknown'))
                    predicted_slots.append(output.get('slots', {}))
            else:
                predicted_intents = ['unknown']
                predicted_slots = [{}]
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            predicted_intents = ['unknown']
            predicted_slots = [{}]
        
        # Handle ground truth - convert to lists if needed
        if isinstance(ground_truth, list):
            # Multiple ground truth frames
            gt_intents = [gt.get('intent', 'unknown') for gt in ground_truth]
            gt_slots = [gt.get('slots', {}) for gt in ground_truth]
        else:
            # Single ground truth frame
            gt_intents = [ground_truth.get('intent', 'unknown')]
            gt_slots = [ground_truth.get('slots', {})]
        
        # Store results
        predicted_intents_list.append(predicted_intents)
        ground_truth_intents_list.append(gt_intents)
        predicted_slots_list.append(predicted_slots)
        ground_truth_slots_list.append(gt_slots)
        
        # Store detailed prediction data
        predictions_data.append({
            'input': user_input,
            'prediction': {
                'intents': predicted_intents,
                'slots': predicted_slots
            },
            'ground_truth': {
                'intents': gt_intents,
                'slots': gt_slots
            }
        })
    
    # Save predictions to file
    if translator:
        predictions_file = f"evaluation/nlu_predictions_translation_{args.model_name}.json"
        if args.avoid_input_translation:
            predictions_file = f"evaluation/nlu_predictions_implicit_{args.model_name}.json"
    else:
        predictions_file = f"evaluation/nlu_predictions_{args.model_name}.json"
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, indent=2, ensure_ascii=False)
    print(f"NLU predictions saved to {predictions_file}")
    
    # Calculate metrics
    intent_metrics = evaluate_intent_classification(predicted_intents_list, ground_truth_intents_list)
    slot_metrics = evaluate_slot_filling(predicted_slots_list, ground_truth_slots_list, 
                                       predicted_intents_list, ground_truth_intents_list)
    
    # Combine all metrics
    results = {
        **intent_metrics,
        **slot_metrics,
        'total_examples': len(test_data)
    }
    
    return results


def evaluate_nlu_from_pipeline(nlu_model, args, translator=None, testset_path="evaluation/nlu_testset.json"):
    """
    Evaluate NLU component when called from the main pipeline.
    """
    # Load test data
    if translator:
        testset_path = "evaluation/nlu_testset_it.json"
    test_data = load_nlu_testset(testset_path)

    # If debug mode is enabled, faster debugging
    if args.debug:
        # test_data = test_data[:5]
        print(f"Debug mode enabled: Using only {len(test_data)} test samples for evaluation")
    
    # Run evaluation
    results = evaluate_nlu(nlu_model, test_data, args, translator)
    
    # Print results
    print_evaluation_results(results)
    
    # Save results to file
    if translator:
        output_file = f"evaluation/nlu_evaluation_results_translation_{args.model_name}.json"
        if args.avoid_input_translation:
            output_file = f"evaluation/nlu_evaluation_results_implicit_{args.model_name}.json"
    else:
        output_file = f"evaluation/nlu_evaluation_results_{args.model_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results


def load_dm_testset(filepath):
    """Load the DM test dataset from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_dm_accuracy(predictions, ground_truths):
    """
    Evaluate dialogue manager using accuracy.
    Each prediction is evaluated separately against the ground truth.
    """
    correct_predictions = 0
    total_predictions = 0
    
    for pred_actions, gt_actions in zip(predictions, ground_truths):
        # Convert to sets for order-independent comparison
        pred_set = set(pred_actions) if isinstance(pred_actions, list) else set()
        pred_list = list(pred_set)
        gt_set = set(gt_actions) if isinstance(gt_actions, list) else set()
        
        # Evaluate each prediction separately
        for pred_action in pred_list:
            total_predictions += 1
            if pred_action in gt_set:
                correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return {
        'dm_accuracy': accuracy,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions
    }


def evaluate_dm(dm_model, test_data, args):
    """
    Evaluate the DM component on the test dataset.
    """
    print(f"Evaluating DM on {len(test_data)} examples...")
    
    predicted_actions_list = []
    ground_truth_actions_list = []
    predictions_data = []  # Store detailed predictions for saving to file
    
    for i, example in enumerate(test_data):
        if i % 50 == 0:
            print(f"Processing example {i}/{len(test_data)}...")
        
        dialogue_state_data = example['dialogue_state']
        ground_truth_actions = example['ground_truth']
        
        # Get DM predictions
        try:
            # Clear and set up dialogue state for this example
            dm_model.dialogue_state.frames.clear()
            # Add initial system turn to conversation history
            dm_model.dialogue_state.add_turn(DEFAULT_SYSTEM_TURNS["INIT"], "system")
            
            # Add the dialogue state as a new frame
            frame_id = 0
            dm_model.dialogue_state.frames[frame_id] = {
                'intent': dialogue_state_data.get('intent'),
                'slots': dialogue_state_data.get('slots', {})
            }
            
            # Get DM output
            dm_output = dm_model.execute()
            
            # Extract predicted actions
            predicted_actions = []
            if 'next_actions' in dm_output and dm_output['next_actions']:
                for action_key, action_value in dm_output['next_actions'].items():
                    if action_value and action_value not in [None, {}, "None", "null"]:
                        predicted_actions.append(action_value)
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            predicted_actions = []
        
        # Store results
        predicted_actions_list.append(predicted_actions)
        ground_truth_actions_list.append(ground_truth_actions)
        
        # Store detailed prediction data
        predictions_data.append({
            'input': dialogue_state_data,
            'prediction': predicted_actions,
            'ground_truth': ground_truth_actions
        })
        
        if args.debug:
            print(f"Example {i}:")
            print(f"  Dialogue State: {dialogue_state_data}")
            print(f"  Predicted Actions: {predicted_actions}")
            print(f"  Ground Truth Actions: {ground_truth_actions}")
            print(f"  Match: {set(predicted_actions) == set(ground_truth_actions)}")
    
    # Save predictions to file
    predictions_file = f"evaluation/dm_predictions_{args.model_name}.json"
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, indent=2, ensure_ascii=False)
    print(f"DM predictions saved to {predictions_file}")
    
    # Calculate metrics
    dm_metrics = evaluate_dm_accuracy(predicted_actions_list, ground_truth_actions_list)
    
    # Combine all metrics
    results = {
        **dm_metrics,
        'total_examples': len(test_data)
    }
    
    return results


def evaluate_dm_from_pipeline(dm_model, args, testset_path="evaluation/dm_testset.json"):
    """
    Evaluate DM component when called from the main pipeline.
    """
    # Load test data
    test_data = load_dm_testset(testset_path)
    
    # If debug mode is enabled, faster debugging
    if args.debug:
        test_data = test_data[:5]
        print(f"Debug mode enabled: Using only {len(test_data)} test samples for DM evaluation")
    
    # Run evaluation
    results = evaluate_dm(dm_model, test_data, args)
    
    # Print results
    print_dm_evaluation_results(results)
    
    # Save results to file
    output_file = f"evaluation/dm_evaluation_results_{args.model_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDM results saved to {output_file}")
    
    return results


def print_dm_evaluation_results(results):
    """Print DM evaluation results in a formatted way."""
    print("\n" + "="*60)
    print("DIALOGUE MANAGER EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nTotal examples evaluated: {results['total_examples']}")
    print(f"Correct predictions: {results['correct_predictions']}")
    print(f"Accuracy: {results['dm_accuracy']:.4f}")
    
    print("="*60)


def print_evaluation_results(results):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*60)
    print("NLU EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nTotal examples evaluated: {results['total_examples']}")
    
    print(f"\nINTENT CLASSIFICATION:")
    print(f"  Accuracy (exact match): {results['intent_accuracy']:.4f}")
    if 'intent_precision' in results:
        print(f"  Precision: {results['intent_precision']:.4f}")
        print(f"  Recall:    {results['intent_recall']:.4f}")
        print(f"  F1-score:  {results['intent_f1']:.4f}")
        print(f"  Correct intents: {results['total_correct']}/{results['total_predicted']} predicted, {results['total_ground_truth']} expected")
    
    print(f"\nSLOT FILLING:")
    if 'valid_slot_evaluations' in results:
        print(f"  Valid slot evaluations (correct intents only): {results['valid_slot_evaluations']}")
    print(f"  Micro-averaged:")
    print(f"    Precision: {results['slot_micro_precision']:.4f}")
    print(f"    Recall:    {results['slot_micro_recall']:.4f}")
    print(f"    F1-score:  {results['slot_micro_f1']:.4f}")
    
    print(f"  Macro-averaged:")
    print(f"    Precision: {results['slot_macro_precision']:.4f}")
    print(f"    Recall:    {results['slot_macro_recall']:.4f}")
    print(f"    F1-score:  {results['slot_macro_f1']:.4f}")
    
    if 'per_slot_metrics' in results and results['per_slot_metrics']:
        print(f"\nPER-SLOT METRICS:")
        for slot, metrics in results['per_slot_metrics'].items():
            print(f"  {slot}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-score:  {metrics['f1']:.4f}")
    
    print("="*60)


def create_chunker_testset(original_testset, num_samples=100):
    """
    Create a test dataset for chunker evaluation by concatenating random pairs of samples.
    Ensures that the two selected samples have different intents.
    """
    chunker_testset = []
    max_attempts = 1000  # Prevent infinite loops
    
    for i in range(num_samples):
        attempts = 0
        sample1 = None
        sample2 = None
        
        # Keep trying until we find two samples with different intents
        while attempts < max_attempts:
            sample1, sample2 = random.sample(original_testset, 2)
            intent1 = sample1["ground_truth"]["intent"]
            intent2 = sample2["ground_truth"]["intent"]
            
            # Break if intents are different
            if intent1 != intent2:
                break
            attempts += 1
        
        # If we couldn't find different intents after max attempts, use what we have
        if attempts >= max_attempts:
            print(f"Warning: Could not find samples with different intents for sample {i}, using samples with same intent")
        
        # Concatenate the user inputs
        concatenated_input = sample1["user_input"] + " " + sample2["user_input"]
        
        # Create ground truth as ordered dictionary of intent-chunk pairs
        intent1 = sample1["ground_truth"]["intent"]
        intent2 = sample2["ground_truth"]["intent"]
        chunk1 = sample1["user_input"]
        chunk2 = sample2["user_input"]
        
        # Create ordered ground truth dictionary maintaining the order of chunks
        ground_truth_chunks = {
            intent1: chunk1,
            intent2: chunk2
        }
        
        chunker_testset.append({
            "user_input": concatenated_input,
            "ground_truth": ground_truth_chunks,
            "original_samples": [sample1, sample2]  # Keep reference to original samples for debugging
        })
    
    return chunker_testset


def evaluate_chunker_classification(predictions, ground_truths):
    """
    Evaluate chunker intent classification using precision, recall, and F1-score.
    Uses set-based comparison - checks if each predicted intent is in the ground truth set.
    """
    total_correct = 0
    total_predicted = 0
    total_ground_truth = 0
    total_exact_matches = 0
    
    for pred_chunks, gt_chunks in zip(predictions, ground_truths):
        # Handle case where prediction failed
        if pred_chunks is None or not isinstance(pred_chunks, dict):
            pred_chunks = {}
        
        # Get sets of intents
        gt_intents = set(gt_chunks.keys()) if isinstance(gt_chunks, dict) else set()
        pred_intents = set(pred_chunks.keys()) if isinstance(pred_chunks, dict) else set()
        
        # Count true positives (correctly predicted intents)
        correct_intents = pred_intents.intersection(gt_intents)
        total_correct += len(correct_intents)
        
        # Count total predictions and ground truths
        total_predicted += len(pred_intents)
        total_ground_truth += len(gt_intents)
        
        # Count exact matches (all intents correct, regardless of order)
        if pred_intents == gt_intents:
            total_exact_matches += 1
    
    # Calculate metrics
    precision = total_correct / total_predicted if total_predicted > 0 else 0.0
    recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = total_exact_matches / len(predictions) if len(predictions) > 0 else 0.0
    
    return {
        'chunker_accuracy': accuracy,
        'chunker_precision': precision,
        'chunker_recall': recall,
        'chunker_f1': f1,
        'total_correct': total_correct,
        'total_predicted': total_predicted,
        'total_ground_truth': total_ground_truth
    }


def evaluate_chunker(nlu_model, test_data, args, translator=None):
    """
    Evaluate the chunker component on the test dataset.
    """
    print(f"Evaluating Chunker on {len(test_data)} examples...")
    
    predicted_chunks_list = []
    ground_truth_chunks_list = []
    predictions_data = []  # Store detailed predictions for saving to file
    
    for i, example in enumerate(test_data):
        if i % 10 == 0:
            print(f"Processing example {i+1}/{len(test_data)}")
        
        user_input = example["user_input"]
        ground_truth = example["ground_truth"]  # Now a dictionary of intent-chunk pairs
        
        # Get chunker prediction
        try:
            predicted_chunks = nlu_model.run_chunker(user_input)
            predicted_chunks_list.append(predicted_chunks)
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            predicted_chunks = {}
            predicted_chunks_list.append(predicted_chunks)
        
        ground_truth_chunks_list.append(ground_truth)
        
        # Store detailed prediction for saving
        predictions_data.append({
            "user_input": user_input,
            "predicted_chunks": predicted_chunks,
            "ground_truth_chunks": ground_truth,
            "original_samples": example.get("original_samples", [])
        })
    
    # Save predictions to file
    if translator:
        predictions_file = f"evaluation/chunker_predictions_it_{args.model_name}.json"
    else:
        predictions_file = f"evaluation/chunker_predictions_{args.model_name}.json"
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, indent=2, ensure_ascii=False)
    print(f"Chunker predictions saved to {predictions_file}")
    
    # Calculate metrics
    chunker_metrics = evaluate_chunker_classification(predicted_chunks_list, ground_truth_chunks_list)
    
    # Combine all metrics
    results = {
        **chunker_metrics,
        'total_examples': len(test_data)
    }
    
    return results


def evaluate_chunker_from_pipeline(nlu_model, args, translator=None, testset_path="evaluation/nlu_testset.json"):
    """
    Evaluate Chunker component when called from the main pipeline.
    """
    # Load original test data
    if translator:
        testset_path = "evaluation/nlu_testset_it.json"
    original_test_data = load_nlu_testset(testset_path)
    
    # Create chunker test dataset by concatenating random pairs
    if args.debug:
        num_samples = 10  # Smaller dataset for debugging
        print(f"Debug mode enabled: Creating {num_samples} chunker test samples")
    else:
        num_samples = 100  # Full dataset as requested
        print(f"Creating {num_samples} chunker test samples by concatenating random pairs...")
    
    chunker_test_data = create_chunker_testset(original_test_data, num_samples)
    
    # Save the created test dataset for reference
    if translator:
        testset_file = f"evaluation/chunker_testset_it_{args.model_name}.json"
    else:
        testset_file = f"evaluation/chunker_testset_{args.model_name}.json"
    os.makedirs(os.path.dirname(testset_file), exist_ok=True)
    with open(testset_file, 'w', encoding='utf-8') as f:
        json.dump(chunker_test_data, f, indent=2, ensure_ascii=False)
    print(f"Chunker test dataset saved to {testset_file}")
    
    # Run evaluation
    results = evaluate_chunker(nlu_model, chunker_test_data, args, translator)
    
    # Print results
    print_chunker_evaluation_results(results)
    
    # Save results to file
    if translator:
        output_file = f"evaluation/chunker_evaluation_results_it_{args.model_name}.json"
    else:
        output_file = f"evaluation/chunker_evaluation_results_{args.model_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nChunker results saved to {output_file}")
    
    return results


def print_chunker_evaluation_results(results):
    """Print Chunker evaluation results in a formatted way."""
    print("\n" + "="*60)
    print("CHUNKER EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nTotal examples evaluated: {results['total_examples']}")
    print(f"Correct predictions: {results['total_correct']}")
    print(f"Total predicted intents: {results['total_predicted']}")
    print(f"Total ground truth intents: {results['total_ground_truth']}")
    
    print(f"\nAccuracy (exact matches): {results['chunker_accuracy']:.4f}")
    print(f"Precision: {results['chunker_precision']:.4f}")
    print(f"Recall: {results['chunker_recall']:.4f}")
    print(f"F1-score: {results['chunker_f1']:.4f}")
    
    print("="*60)

