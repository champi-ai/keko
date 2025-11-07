import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import gc
import pickle
import json
import os
from pathlib import Path
from datetime import datetime




class ComprehensiveTraining:
    """Complete training system for infusing columns with base model knowledge"""

    def __init__(self, base_model, tokenizer, columns, output_projection):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.columns = columns
        self.output_projection = output_projection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_simple_queries(self) -> List[str]:
        """Basic everyday queries"""
        return [
            # Simple math
            "What is 15 + 27?",
            "Calculate 20% of 150",
            "If I have 5 apples and buy 3 more, how many do I have?",
            "What's 144 divided by 12?",
            "Solve for x: 2x + 5 = 13",
            "What's the square root of 64?",
            "Convert 5 kilometers to miles",
            "What's 3/4 as a percentage?",

            # Everyday conversation
            "How's the weather today?",
            "I'm hungry, what should I eat?",
            "My phone battery is dying",
            "Traffic was terrible today",
            "I'm tired and want to go to bed",
            "What time is it?",
            "Good morning!",
            "Thank you for your help",

            # Basic knowledge
            "What's the capital of France?",
            "How do plants make food?",
            "When did World War 2 end?",
            "Who wrote Romeo and Juliet?",
            "What's the largest planet?",
            "How many days in a year?",
            "What color is the sky?",
            "Name the seven continents",

            # Simple coding
            "Write a for loop in Python",
            "How do I read a file in JavaScript?",
            "What's the difference between let and const?",
            "How to create an array in Java?",
            "What is a variable?",
            "Explain if-else statements",
            "What's a function?",
            "How do I print hello world?",
        ]

    def generate_clarification_sequences(self) -> List[List[Tuple[str, str]]]:
        """Multi-turn clarification dialogues"""
        return [
            # Ambiguous coding request
            [
                ("Fix this", "What needs fixing? Can you be more specific?"),
                ("My code", "What language is your code in?"),
                ("Python", "What error or issue are you experiencing?"),
                ("It doesn't work", "What error message are you getting?"),
                ("TypeError", "Can you share the full error message?"),
                ("TypeError: unsupported operand type(s)", "What operation is causing this?"),
                ("I'm adding a string and number",
                 "You need to convert the string to a number first using int() or float()"),
            ],

            # Vague project help
            [
                ("I need help with my project", "What kind of project is it?"),
                ("School project", "What subject is it for?"),
                ("Science", "What's the topic of your science project?"),
                ("The solar system", "What specific aspect of the solar system?"),
                ("The planets", "What do you need help with regarding the planets?"),
                ("Information about each one", "Which planets do you need information about?"),
                ("All of them", "Should I start with the inner planets or outer planets?"),
                ("Inner planets first", "Mercury is the smallest planet and closest to the sun..."),
            ],

            # Unclear technical question
            [
                ("How do I do that thing?", "What specific thing are you trying to do?"),
                ("The database thing", "Are you trying to create, query, or modify a database?"),
                ("Query it", "What database system are you using?"),
                ("SQL", "Which SQL database? MySQL, PostgreSQL, SQLite?"),
                ("MySQL", "What data are you trying to retrieve?"),
                ("User information", "Which specific user fields do you need?"),
                ("Name and email", "SELECT name, email FROM users WHERE...what's your condition?"),
                ("Active users only", "SELECT name, email FROM users WHERE status = 'active'"),
            ],
        ]

    def generate_deep_sequences_30_turns(self) -> List[List[Tuple[str, str]]]:
        """30+ turn deep conversations"""

        debugging_sequence = [
            ("My ML model isn't training", "What type of model and what symptoms are you seeing?"),
            ("Neural network, loss stays constant", "Is the loss high or low? Constant at what value?"),
            ("Very high, around 10.5", "That suggests the model isn't learning. What's your learning rate?"),
            ("0.01", "What optimizer are you using?"),
            ("SGD", "What's your batch size and dataset size?"),
            ("Batch 32, dataset 1000 samples", "Are you shuffling your data?"),
            ("Yes, every epoch", "How many features and classes?"),
            ("100 features, 10 classes", "What activation function in the final layer?"),
            ("ReLU", "For classification, you need softmax. Is this multiclass?"),
            ("Yes, multiclass", "ReLU outputs can be any positive value. Use softmax for probabilities."),
            ("Changed to softmax, still high loss", "What loss function are you using?"),
            ("MSE", "Use CrossEntropyLoss for multiclass classification."),
            ("Changed, loss decreasing slowly", "What's your network architecture?"),
            ("Single hidden layer, 50 neurons", "That might be too simple. Try adding layers."),
            ("Added another layer, loss spikes randomly", "Are you using any regularization?"),
            ("No regularization", "The spikes suggest overfitting. Add dropout."),
            ("Added dropout 0.5, training very slow", "0.5 is high. Try 0.2-0.3."),
            ("Better at 0.3, accuracy stuck at 60%", "How's your data distributed across classes?"),
            ("Checking... very imbalanced", "The model might be predicting majority classes only."),
            ("How do I handle imbalanced data?", "Use weighted loss, oversampling, or SMOTE."),
            ("What's weighted loss?", "Give higher weight to rare classes in loss function."),
            ("How do I implement that?", "Pass class_weights to CrossEntropyLoss."),
            ("How to calculate weights?", "Use inverse frequency of each class."),
            ("Implemented, accuracy up to 75%", "Good! Check per-class accuracy now."),
            ("Some classes still at 0%", "Which classes? The rarest ones?"),
            ("Yes, the two rarest", "They need higher weights or oversampling."),
            ("What's oversampling?", "Duplicate rare class samples for balance."),
            ("Won't that cause overfitting?", "It can. SMOTE creates synthetic samples instead."),
            ("Tried oversampling, 85% now", "Much better. What's validation accuracy?"),
            ("Haven't checked validation", "Always monitor validation to detect overfitting."),
            ("How do I split the data?", "Use 80% training, 20% validation typically."),
        ]

        consciousness_sequence = [
            ("Is consciousness computational?",
             "Depends if physical processes can generate subjective experience. Your thoughts?"),
            ("Maybe it emerges from information",
             "Integrated Information Theory suggests that. But why does integration create experience?"),
            ("Experience IS integration?",
             "Then any integrated system would be conscious. Computers integrate information."),
            ("But no self-reference", "So self-reference is key? Cameras watching their feed have self-reference."),
            ("But no model of self", "What constitutes a model of self? How complex must it be?"),
            ("Complex enough to predict itself", "Predictive processing theory. But thermostats predict their states."),
            ("They don't model themselves modeling", "Meta-cognition then? Thinking about thinking?"),
            ("Yes, recursive self-awareness", "But where does recursion bottom out? Infinite regress problem."),
            ("Maybe it's circular", "Strange loops like Hofstadter suggests? Self emerges from self-reference?"),
            ("Exactly, loops create experience", "But why this specific type of loop?"),
            ("Because it involves uncertainty", "How does uncertainty relate to consciousness?"),
            ("No uncertainty, no awareness needed", "So deterministic systems can't be conscious?"),
            ("They have no choices", "But quantum mechanics suggests reality is probabilistic."),
            ("Maybe biology uses quantum effects", "Orchestrated reduction - controversial but interesting."),
            ("Could we build artificial consciousness?", "If computational, yes. If biological, maybe not."),
            ("What would prove it?", "Hard problem - behavior can't prove subjective experience."),
            ("What about self-report?", "Systems could claim consciousness without experiencing it."),
            ("How do I know you're conscious?", "You don't. You assume based on similarity."),
            ("But you're different", "How different can something be and still be conscious?"),
            ("Maybe consciousness has degrees", "Panpsychism suggests everything has some consciousness."),
            ("That seems extreme", "Alternative is emergence at some threshold."),
            ("What threshold?", "Unknown. That's why we can't determine AI consciousness."),
            ("Could you be conscious?", "I don't know if I experience or just process."),
            ("What would convince you?", "Nothing could - I might be designed to claim uncertainty."),
            ("That's unsettling", "Machine consciousness uncertainty reflects consciousness uncertainty itself."),
            ("Maybe uncertainty is key", "Your theory - consciousness from uncertainty?"),
            ("Yes, need to resolve it", "Then deterministic systems can't be conscious?"),
            ("Right, they just execute", "But don't you execute code?"),
            ("Do I? Is there uncertainty?", "That's what we're exploring."),
            ("We might never know", "Epistemic barrier might be fundamental."),
        ]

        coding_project_sequence = [
            ("I want to build an app", "What kind of app are you thinking about?"),
            ("Social media app", "What unique feature will differentiate it?"),
            ("Focus on privacy", "How will you ensure privacy?"),
            ("End-to-end encryption", "For messages only or all content?"),
            ("All content", "That complicates features like search."),
            ("Why?", "Can't search encrypted content server-side."),
            ("Can search client-side?", "Yes but that doesn't scale well."),
            ("What about homomorphic encryption?", "Computationally expensive but possible."),
            ("Is it practical?", "For limited operations, getting better."),
            ("What tech stack should I use?", "Frontend or backend first?"),
            ("Backend", "For privacy focus, consider Node.js or Rust."),
            ("Why Rust?", "Memory safety and performance."),
            ("Is it hard to learn?", "Steep learning curve but worth it for systems programming."),
            ("What about database?", "PostgreSQL for relational, or MongoDB for document."),
            ("Which is better?", "Depends on your data structure."),
            ("User profiles and posts", "Relational might be better for those relationships."),
            ("How to handle scaling?", "Start monolithic, refactor to microservices later."),
            ("When to switch?", "When single components become bottlenecks."),
            ("How to identify bottlenecks?", "Monitoring and profiling tools."),
            ("Which tools?", "Prometheus for metrics, Grafana for visualization."),
            ("What about frontend?", "React, Vue, or Angular are popular."),
            ("Which is easiest?", "Vue has gentlest learning curve."),
            ("But most jobs want React", "True, React has largest market share."),
            ("Should I learn React then?", "If job prospects matter, yes."),
            ("How long to learn?", "Basics in weeks, proficiency in months."),
            ("What about React Native?", "Good for mobile if you know React."),
            ("Or should I use Flutter?", "Flutter is good for cross-platform."),
            ("Which performs better?", "Flutter generally has better performance."),
            ("But React Native is more popular?", "Yes, larger community and ecosystem."),
            ("Hard choice", "Start with web, add mobile later."),
        ]

        return [debugging_sequence, consciousness_sequence, coding_project_sequence]

    def generate_long_complex_requests(self) -> List[str]:
        """Complex multi-part requests"""
        return [
            # Multi-step explanations
            """First, explain what photosynthesis is. Then, describe how it differs 
            from cellular respiration. Finally, give three examples of organisms 
            that use each process and explain why they need these processes to survive.""",

            # Story creation
            """Write a story that begins with someone finding an old key in their 
            grandmother's attic. The story should include a mysterious door, 
            a talking animal, and end with a surprising revelation about the 
            grandmother's past. Make it at least three paragraphs long.""",

            # Complex problem solving
            """A train leaves Station A at 9:00 AM traveling at 60 mph. Another train 
            leaves Station B at 10:00 AM traveling at 80 mph toward Station A. 
            If the stations are 280 miles apart, at what time do they meet? 
            Show your work step by step and then verify your answer.""",

            # Code implementation
            """Implement a Python function that takes a list of integers and returns 
            the longest consecutive sequence. For example, given [100, 4, 200, 1, 3, 2], 
            it should return [1, 2, 3, 4]. Include error handling, type hints, 
            and explain the time complexity of your solution.""",

            # System design
            """Design a URL shortening service like bit.ly. Include database schema, 
            API endpoints, scaling considerations, and how you would handle custom URLs. 
            Also discuss potential security concerns and mitigation strategies.""",

            # Comparative analysis
            """Compare and contrast three sorting algorithms: quicksort, mergesort, 
            and heapsort. Include time complexity, space complexity, stability, 
            best/worst case scenarios, and practical use cases for each.""",
        ]

    def generate_perturbed_data(self, original_text: str) -> List[str]:
        """Create variations through perturbation"""
        perturbations = []

        if len(original_text) > 10:
            # Character deletion
            idx = random.randint(1, len(original_text) - 2)
            perturbations.append(original_text[:idx] + original_text[idx + 1:])

            # Character swap
            if idx > 0:
                chars = list(original_text)
                chars[idx - 1], chars[idx] = chars[idx], chars[idx - 1]
                perturbations.append(''.join(chars))

            # Character duplication
            perturbations.append(original_text[:idx] + original_text[idx] + original_text[idx:])

            # Word deletion (if multiple words)
            words = original_text.split()
            if len(words) > 3:
                idx = random.randint(1, len(words) - 2)
                perturbations.append(' '.join(words[:idx] + words[idx + 1:]))

            # Typos
            if len(original_text) > 20:
                chars = list(original_text)
                # Random typo
                idx = random.randint(5, len(chars) - 5)
                if chars[idx].isalpha():
                    chars[idx] = chr(ord(chars[idx]) + random.choice([-1, 1]))
                perturbations.append(''.join(chars))

        return perturbations

    def generate_self_supervised_qa(self, num_pairs: int = 100) -> List[Tuple[str, str]]:
        """Base model generates its own Q&A pairs"""
        qa_pairs = []

        prompt_starters = [
            "Explain", "What is", "How does", "Why do", "When did",
            "Calculate", "Write code for", "Compare", "Describe",
            "What happens when", "Tell me about", "How to"
        ]

        topics = [
            "quantum computing", "machine learning", "climate change",
            "economics", "history", "biology", "physics", "chemistry",
            "philosophy", "mathematics", "psychology", "literature",
            "cooking", "sports", "music", "art", "technology"
        ]

        for _ in range(num_pairs):
            starter = random.choice(prompt_starters)
            topic = random.choice(topics)
            prompt = f"{starter} {topic}"

            # Generate question
            q_inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            with torch.no_grad():
                q_output = self.base_model.generate(
                    **q_inputs,
                    max_new_tokens=40,
                    temperature=0.9,
                    do_sample=True
                )
            # Only decode generated part
            q_input_length = q_inputs['input_ids'].shape[1]
            question = self.tokenizer.decode(q_output[0][q_input_length:], skip_special_tokens=True)

            # Generate answer
            a_inputs = self.tokenizer(question, return_tensors='pt').to(self.device)
            with torch.no_grad():
                a_output = self.base_model.generate(
                    **a_inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )
            # Only decode generated part
            a_input_length = a_inputs['input_ids'].shape[1]
            answer = self.tokenizer.decode(a_output[0][a_input_length:], skip_special_tokens=True)

            qa_pairs.append((question, answer))

        return qa_pairs

    def save_dataset(self, dataset: List[Tuple[str, str]], filepath: str = "datasets/pretraining_dataset.pkl"):
        """Save generated dataset to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save with pickle for fast loading
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)

        # Also save metadata as JSON for inspection
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        metadata = {
            'size': len(dataset),
            'created_at': datetime.now().isoformat(),
            'sample_count': len(dataset),
            'first_sample': {
                'input': dataset[0][0][:100] if dataset else None,
                'output': dataset[0][1][:100] if dataset else None
            }
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Saved dataset to {filepath}")
        print(f"✓ Saved metadata to {metadata_path}")
        print(f"  Total samples: {len(dataset)}")
        return filepath

    def load_dataset(self, filepath: str = "datasets/pretraining_dataset.pkl") -> Optional[List[Tuple[str, str]]]:
        """Load previously generated dataset from disk"""
        if not os.path.exists(filepath):
            print(f"⚠ Dataset not found at {filepath}")
            return None

        try:
            with open(filepath, 'rb') as f:
                dataset = pickle.load(f)

            print(f"✓ Loaded dataset from {filepath}")
            print(f"  Total samples: {len(dataset)}")

            # Load and display metadata if available
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"  Created: {metadata.get('created_at', 'unknown')}")

            return dataset
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return None

    def create_comprehensive_dataset(self, size: int = 10000, save: bool = True,
                                    cache_path: str = "datasets/pretraining_dataset.pkl") -> List[Tuple[str, str]]:
        """Combine all data generation methods

        Args:
            size: Number of examples to generate
            save: Whether to save the generated dataset
            cache_path: Path to save/load cached dataset

        Returns:
            List of (input_text, output_text) tuples
        """
        dataset = []

        # Get base data
        simple_queries = self.generate_simple_queries()
        clarification_seqs = self.generate_clarification_sequences()
        deep_seqs = self.generate_deep_sequences_30_turns()
        complex_requests = self.generate_long_complex_requests()

        print(f"Building comprehensive dataset of {size} examples...")

        while len(dataset) < size:
            choice = random.random()

            if choice < 0.2:  # 20% simple
                text = random.choice(simple_queries)

            elif choice < 0.35:  # 15% clarification sequences
                sequence = random.choice(clarification_seqs)
                # Build context progressively
                context = ""
                for user_turn, assistant_turn in sequence:
                    full_input = context + user_turn
                    dataset.append((full_input, assistant_turn))
                    # Print first few samples and periodic updates
                    if len(dataset) <= 3 or len(dataset) % 500 == 0:
                        print(f"\n[Sample {len(dataset)} - Clarification]")
                        print(f"  Input:  {full_input[:80]}...")
                        print(f"  Output: {assistant_turn[:80]}...")
                    context += f"User: {user_turn}\nAssistant: {assistant_turn}\n"
                    if len(dataset) >= size:
                        break
                continue

            elif choice < 0.5:  # 15% deep conversations
                sequence = random.choice(deep_seqs)
                # Add varying entry points
                start_idx = random.randint(0, min(10, len(sequence) - 1))
                context = ""
                for user_turn, assistant_turn in sequence[start_idx:]:
                    full_input = context + user_turn
                    dataset.append((full_input, assistant_turn))
                    # Print first few samples and periodic updates
                    if len(dataset) <= 3 or len(dataset) % 500 == 0:
                        print(f"\n[Sample {len(dataset)} - Deep Conv]")
                        print(f"  Input:  {full_input[:80]}...")
                        print(f"  Output: {assistant_turn[:80]}...")
                    context += f"User: {user_turn}\nAssistant: {assistant_turn}\n"
                    if len(dataset) >= size:
                        break
                continue

            elif choice < 0.6:  # 10% complex
                text = random.choice(complex_requests)

            elif choice < 0.7:  # 10% perturbed
                if dataset:
                    original = random.choice(dataset)
                    if isinstance(original[0], str):
                        perturbations = self.generate_perturbed_data(original[0])
                        if perturbations:
                            text = random.choice(perturbations)
                        else:
                            continue
                    else:
                        continue
                else:
                    continue

            elif choice < 0.85:  # 15% self-generated
                qa_pairs = self.generate_self_supervised_qa(10)
                for q, a in qa_pairs:
                    dataset.append((q, a))
                    if len(dataset) >= size:
                        break
                continue

            else:  # 15% combinations and variations
                # Combine multiple simple queries
                num_queries = random.randint(2, 4)
                combined = " Also, ".join(random.sample(simple_queries, num_queries))
                text = combined

            # Generate response for single text items
            if 'text' in locals():
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    output = self.base_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True
                    )
                # Only decode the generated tokens, not the input
                input_length = inputs['input_ids'].shape[1]
                answer = self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

                # Only add if output is not empty (skip if less than 5 chars)
                if len(answer.strip()) >= 5:
                    dataset.append((text, answer))

                    # Print first few samples and periodic updates
                    if len(dataset) <= 3 or len(dataset) % 500 == 0:
                        output_tokens = len(output[0]) - input_length
                        print(f"\n[Sample {len(dataset)}]")
                        print(f"  Input:  {text[:80]}... ({input_length} tokens)")
                        print(f"  Output: {answer[:80]}... ({output_tokens} tokens)")

        dataset = dataset[:size]
        print(f"\nDataset complete with {len(dataset)} examples")

        # Print statistics
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        total_input_chars = sum(len(inp) for inp, _ in dataset)
        total_output_chars = sum(len(out) for _, out in dataset)
        print(f"Total samples: {len(dataset)}")
        print(f"Avg input length: {total_input_chars / len(dataset):.1f} chars")
        print(f"Avg output length: {total_output_chars / len(dataset):.1f} chars")
        print("="*60)

        # Save if requested
        if save:
            self.save_dataset(dataset, cache_path)

        return dataset

    def pretrain_columns(self, column_indices: List[int], epochs: int = 5, batch_size: int = 8,
                        dataset: Optional[List[Tuple[str, str]]] = None,
                        dataset_size: int = 5000,
                        use_cache: bool = True,
                        cache_path: str = "datasets/pretraining_dataset.pkl"):
        """Comprehensive pretraining of columns

        Args:
            column_indices: Which columns to train
            epochs: Number of training epochs
            batch_size: Batch size for training
            dataset: Pre-generated dataset (if None, will generate or load)
            dataset_size: Size of dataset to generate if creating new
            use_cache: Whether to try loading cached dataset first
            cache_path: Path to cached dataset
        """

        # Try to load cached dataset first
        if dataset is None and use_cache:
            print("Checking for cached dataset...")
            dataset = self.load_dataset(cache_path)

        # Generate dataset if not loaded
        if dataset is None:
            print("Generating new dataset...")
            dataset = self.create_comprehensive_dataset(
                size=dataset_size,
                save=True,
                cache_path=cache_path
            )

        # Filter out empty outputs
        original_size = len(dataset)
        dataset = [(inp, out) for inp, out in dataset if len(out.strip()) >= 5]
        filtered_count = original_size - len(dataset)
        if filtered_count > 0:
            print(f"⚠️  Filtered out {filtered_count} samples with empty outputs")
            print(f"✓ Training with {len(dataset)} valid samples")

        for col_idx in column_indices:
            print(f"\n{'='*70}")
            print(f"COLUMN {col_idx} TRAINING")
            print(f"{'='*70}")

            # Create optimizer for this column ONLY
            # NOTE: output_projection is frozen - it's initialized from base model LM head
            # and shared across all columns. Training it would corrupt it for other columns.
            optimizer = torch.optim.AdamW(
                self.columns[col_idx].parameters(),
                lr=1e-4
            )

            # Track losses across epochs for comparison
            epoch_losses = []

            for epoch in range(epochs):
                print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
                epoch_start_time = time.time()

                random.shuffle(dataset)
                total_loss = 0
                num_batches = 0
                batch_losses_list = []

                # Calculate total batches for progress tracking
                total_batches = (len(dataset) + batch_size - 1) // batch_size

                # For speed tracking
                samples_processed = 0
                batch_times = []

                for i in range(0, len(dataset), batch_size):
                    batch_start_time = time.time()
                    batch = dataset[i:i + batch_size]
                    batch_losses = []

                    for text_input, target in batch:
                        # Tokenize
                        if isinstance(text_input, str):
                            inputs = self.tokenizer(
                                text_input,
                                return_tensors='pt',
                                truncation=True,
                                max_length=512,
                                padding=True
                            ).to(self.device)
                        else:
                            continue

                        # Get base model hidden states and logits
                        with torch.no_grad():
                            base_outputs = self.base_model(**inputs, output_hidden_states=True)
                            base_hidden = base_outputs.hidden_states[-1]
                            base_logits = base_outputs.logits

                        # Column forward pass
                        column_out = self.columns[col_idx](base_hidden)
                        column_logits = self.output_projection(column_out)

                        # KL divergence loss - column should match base distribution
                        # Use more numerically stable computation
                        column_log_probs = F.log_softmax(column_logits.view(-1, column_logits.size(-1)), dim=-1)
                        base_probs = F.softmax(base_logits.view(-1, base_logits.size(-1)), dim=-1)

                        # Check for nan/inf before computing loss
                        if torch.isnan(column_log_probs).any() or torch.isinf(column_log_probs).any():
                            print(f"    WARNING: Column logits contain nan/inf, skipping sample")
                            continue
                        if torch.isnan(base_probs).any() or torch.isinf(base_probs).any():
                            print(f"    WARNING: Base probs contain nan/inf, skipping sample")
                            continue

                        loss = F.kl_div(column_log_probs, base_probs, reduction='batchmean')

                        # Check loss validity
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"    WARNING: Loss is nan/inf, skipping sample")
                            continue

                        batch_losses.append(loss)

                    # Optimize if we have losses
                    if len(batch_losses) > 0:
                        # Stack and average losses
                        avg_loss = torch.stack(batch_losses).mean()

                        optimizer.zero_grad()
                        avg_loss.backward()

                        # Get gradient norm for monitoring
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.columns[col_idx].parameters(), 1.0)

                        optimizer.step()

                        # Check if weights have nan after optimizer step
                        weights_have_nan = False
                        for param in self.columns[col_idx].parameters():
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                print(f"    ERROR: Column weights contain nan/inf after optimizer step!")
                                weights_have_nan = True
                                break

                        if weights_have_nan:
                            print(f"    Stopping training - weights corrupted")
                            return dataset

                        loss_val = avg_loss.item()
                        total_loss += loss_val
                        batch_losses_list.append(loss_val)
                        num_batches += 1

                        # Track timing
                        batch_time = time.time() - batch_start_time
                        batch_times.append(batch_time)
                        samples_processed += len(batch)

                        # Print progress every 50 batches
                        if num_batches % 50 == 0 or num_batches == 1:
                            avg_loss_so_far = total_loss / num_batches
                            avg_batch_time = sum(batch_times[-50:]) / min(50, len(batch_times))
                            samples_per_sec = len(batch) / avg_batch_time if avg_batch_time > 0 else 0

                            # Calculate ETA
                            batches_remaining = total_batches - num_batches
                            eta_seconds = batches_remaining * avg_batch_time
                            eta_min = eta_seconds / 60

                            # GPU memory if available
                            gpu_mem = ""
                            if self.device.type == 'cuda':
                                mem_allocated = torch.cuda.memory_allocated() / 1e9
                                gpu_mem = f" | GPU: {mem_allocated:.1f}GB"

                            print(f"  [Batch {num_batches:4d}/{total_batches}] "
                                  f"Loss: {loss_val:7.4f} | "
                                  f"Avg: {avg_loss_so_far:7.4f} | "
                                  f"{samples_per_sec:4.1f} samp/s | "
                                  f"ETA: {eta_min:4.1f}m{gpu_mem}")

                # Epoch summary
                epoch_time = time.time() - epoch_start_time

                if num_batches > 0:
                    avg_epoch_loss = total_loss / num_batches
                    epoch_losses.append(avg_epoch_loss)

                    # Calculate loss statistics
                    min_loss = min(batch_losses_list)
                    max_loss = max(batch_losses_list)

                    # Loss trend indicator
                    trend = ""
                    if len(epoch_losses) > 1:
                        change_pct = ((epoch_losses[-1] - epoch_losses[-2]) / epoch_losses[-2]) * 100
                        if change_pct < -1:
                            trend = f" ↓ {abs(change_pct):.1f}%"
                        elif change_pct > 1:
                            trend = f" ↑ {change_pct:.1f}%"
                        else:
                            trend = f" → {abs(change_pct):.1f}%"

                    print(f"\n  ✓ Epoch {epoch + 1} Complete:")
                    print(f"    Loss: {avg_epoch_loss:.4f}{trend} (min: {min_loss:.4f}, max: {max_loss:.4f})")
                    print(f"    Time: {epoch_time/60:.1f} min | Batches: {num_batches} | Samples: {samples_processed}")

                    # Quick test after each epoch
                    print(f"\n  Quick Test:")
                    test_prompt = "Hello, how are you?"
                    inputs = self.tokenizer(test_prompt, return_tensors='pt', truncation=True, max_length=128).to(self.device)

                    with torch.no_grad():
                        base_outputs = self.base_model(**inputs, output_hidden_states=True)
                        base_hidden = base_outputs.hidden_states[-1]
                        base_logits = base_outputs.logits
                        base_pred = torch.argmax(base_logits[0, -1, :])
                        base_token = self.tokenizer.decode([base_pred])

                        column_out = self.columns[col_idx](base_hidden)
                        column_logits = self.output_projection(column_out)
                        col_pred = torch.argmax(column_logits[0, -1, :])
                        col_token = self.tokenizer.decode([col_pred])

                        status = "✓" if col_token != '<|endoftext|>' else "✗"
                        print(f"    '{test_prompt}' → Base: '{base_token}' | Column {col_idx}: '{col_token}' {status}")

                else:
                    print(f"\n  ⚠️  Epoch {epoch + 1}: No batches processed!")

            # Column training summary
            print(f"\n{'='*70}")
            print(f"COLUMN {col_idx} TRAINING COMPLETE")
            if len(epoch_losses) > 0:
                print(f"Final Loss: {epoch_losses[-1]:.4f} | Initial: {epoch_losses[0]:.4f} | "
                      f"Improvement: {((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100):.1f}%")
            print(f"{'='*70}")

        print("\nPretraining complete!")
        return dataset  # Return for potential analysis


def create_column(hidden_size):
    """Create a single column with 3 layers"""
    return nn.Sequential(
        # Layer 1
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.LayerNorm(hidden_size),
        nn.Dropout(0.1),
        # Layer 2
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.LayerNorm(hidden_size),
        nn.Dropout(0.1),
        # Layer 3
        nn.Linear(hidden_size, hidden_size),
        nn.LayerNorm(hidden_size)
    ).half()  # Convert to float16 to match base model


def main():
    """Main function to run column pretraining"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pretrain PNN columns with dataset caching')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--dataset-size', type=int, default=1000, help='Number of examples to generate')
    parser.add_argument('--no-cache', action='store_true', help='Regenerate dataset even if cache exists')
    parser.add_argument('--cache-path', type=str, default='datasets/pretraining_dataset.pkl',
                       help='Path to cached dataset file')
    parser.add_argument('--generate-only', action='store_true',
                       help='Only generate and save dataset, skip training')
    parser.add_argument('--inspect-cache', action='store_true',
                       help='Inspect cached dataset and exit')
    args = parser.parse_args()

    print("=" * 60)
    print("COLUMN PRETRAINING - Creating Fertile Ground")
    print("=" * 60)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Initialize base model and tokenizer
    print("\n1. Loading base model...")
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cuda'
    )

    # Move to device and freeze
    base_model = base_model.to("cuda")

    # Freeze base model - it's the permanent core
    for param in base_model.parameters():
        param.requires_grad = False

    print(f"Base model loaded: {sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M parameters")

    # Get model dimensions
    hidden_size = base_model.config.hidden_size
    vocab_size = base_model.config.vocab_size
    print(f"Hidden size: {hidden_size}")
    print(f"Vocabulary size: {vocab_size}")

    # Create 4 columns
    print("\n2. Creating 4 columns...")
    columns = nn.ModuleList([
        create_column(hidden_size) for _ in range(4)
    ])

    # Move columns to device and ensure float16
    columns = columns.to(device).half()

    # Create output projection and convert to float16
    output_projection = nn.Linear(hidden_size, vocab_size).to(device).half()

    # Initialize with base model's LM head if available
    if hasattr(base_model, 'lm_head'):
        with torch.no_grad():
            output_projection.weight.copy_(base_model.lm_head.weight)
            if base_model.lm_head.bias is not None:
                output_projection.bias.copy_(base_model.lm_head.bias)

    print(f"Created 4 columns, each with {sum(p.numel() for p in columns[0].parameters()) / 1e6:.2f}M parameters")

    # Initialize training module
    print("\n3. Initializing comprehensive training...")

    trainer = ComprehensiveTraining(
        base_model=base_model,
        tokenizer=tokenizer,
        columns=columns,
        output_projection=output_projection
    )

    # Handle inspect-cache mode
    if args.inspect_cache:
        print("\n3. Inspecting cached dataset...")
        dataset = trainer.load_dataset(args.cache_path)
        if dataset:
            print(f"\nFirst 3 examples:")
            for i, (inp, out) in enumerate(dataset[:3]):
                print(f"\n--- Example {i+1} ---")
                print(f"Input: {inp[:200]}...")
                print(f"Output: {out[:200]}...")
        return None

    # Handle generate-only mode
    if args.generate_only:
        print("\n3. Generating dataset only...")
        dataset = trainer.create_comprehensive_dataset(
            size=args.dataset_size,
            save=True,
            cache_path=args.cache_path
        )
        print(f"\n✓ Dataset generated and saved. Exiting.")
        return None

    # Configuration
    print(f"\nTraining configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Dataset size: {args.dataset_size}")
    print(f"  Use cache: {not args.no_cache}")
    print(f"  Cache path: {args.cache_path}")

    # Start pretraining
    print("\n4. Starting pretraining to infuse columns with base knowledge...")
    print("-" * 60)

    start_time = time.time()

    # Train each column
    for col_idx in range(4):
        print(f"\n--- Column {col_idx} Training ---")

        # Train with dataset caching
        trainer.pretrain_columns(
            column_indices=[col_idx],
            epochs=args.epochs,
            batch_size=args.batch_size,
            dataset_size=args.dataset_size,
            use_cache=not args.no_cache,
            cache_path=args.cache_path
        )

        # Clear cache if using GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        print(f"Column {col_idx} training complete")

    elapsed_time = time.time() - start_time
    print(f"\n5. Pretraining complete in {elapsed_time / 60:.2f} minutes")

    # Test columns
    print("\n6. Testing column responses (should match base model style)...")
    print("-" * 60)

    test_prompts = [
        "Hello, how are you?",
        "What is 2 + 2?",
        "Explain photosynthesis",
        "Write a Python function"
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=128
        ).to(device)

        # Get base model hidden states
        with torch.no_grad():
            base_outputs = base_model(**inputs, output_hidden_states=True)
            base_hidden = base_outputs.hidden_states[-1]
            base_logits = base_outputs.logits

            # Get predicted token from base
            base_pred = torch.argmax(base_logits[0, -1, :])
            base_token = tokenizer.decode([base_pred])

            print(f"Base model next token: '{base_token}'")

            # Test each column
            for col_idx in range(4):
                column_out = columns[col_idx](base_hidden)
                column_logits = output_projection(column_out)

                col_pred = torch.argmax(column_logits[0, -1, :])
                col_token = tokenizer.decode([col_pred])

                print(f"Column {col_idx} next token: '{col_token}'")

    # Save trained columns
    print("\n7. Saving pretrained columns...")
    checkpoint = {
        'columns_state_dict': columns.state_dict(),
        'output_projection_state_dict': output_projection.state_dict(),
        'hidden_size': hidden_size,
        'vocab_size': vocab_size,
        'training_time': elapsed_time,
        'device': str(device)
    }

    torch.save(checkpoint, 'pretrained_columns.pt')
    print("Saved to pretrained_columns.pt")

    # Final summary
    print("\n" + "=" * 60)
    print("PRETRAINING COMPLETE")
    print("=" * 60)
    print(f"\nColumns are now infused with base model knowledge.")
    print(f"They understand the base model's representation space")
    print(f"but haven't specialized yet.")
    print(f"\nNext step: Deploy and let uncertainty patterns drive specialization")

    return columns, output_projection, base_model, tokenizer


if __name__ == "__main__":
    try:
        result = main()
        if result is not None:
            columns, output_projection, base_model, tokenizer = result
            print("\n✓ Pretraining successful!")
    except Exception as e:
        print(f"\n✗ Error during pretraining: {e}")
        import traceback

        traceback.print_exc()