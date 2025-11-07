import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import deque
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class UncertaintyEvent:
    timestamp: float
    input_pattern: torch.Tensor
    uncertainty_mask: torch.Tensor
    column_response: int
    resolved: bool


class UncertaintyPNN:
    def __init__(self, base_model_name="HuggingFaceTB/SmolLM-360M-Instruct"):
        print("Initializing Uncertainty-Driven PNN...")

        # Base model - frozen core
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base = AutoModelForCausalLM.from_pretrained(base_model_name)

        # Freeze base - it's the permanent core
        for p in self.base.parameters():
            p.requires_grad = False

        self.hidden_size = self.base.config.hidden_size
        self.vocab_size = self.base.config.vocab_size

        # 4 columns for specialization
        self.columns = nn.ModuleList([
            self._create_column() for _ in range(4)
        ])

        # Column states: inactive -> active -> frozen -> homogeneous
        self.column_states = ['active'] * 4
        self.column_scores = [0.0] * 4  # Track complementarity

        # Uncertainty thresholds (x < p < y)
        self.uncertainty_lower = 0.3
        self.uncertainty_upper = 0.7

        # Token hunger state
        self.satisfaction_threshold = 0.8
        self.current_satisfaction = 0.0
        self.token_buffer = deque(maxlen=1000)

        # Background training queue
        self.training_queue = deque(maxlen=10000)
        self.uncertainty_history = deque(maxlen=1000)

        # Frozen column indices (part of extended core)
        self.frozen_columns = []

        # Output projection for columns
        self.output_projection = nn.Linear(self.hidden_size, self.vocab_size)

        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base = self.base.to(self.device)
        self.columns = self.columns.to(self.device)
        self.output_projection = self.output_projection.to(self.device)

    def _create_column(self):
        """Create a single column with 3 layers"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

    def detect_uncertainty(self, logits):
        """Identify tokens with uncertainty (between thresholds)"""
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values

        uncertain = (max_probs > self.uncertainty_lower) & \
                    (max_probs < self.uncertainty_upper)

        return uncertain, max_probs

    def pretrain_fertile_ground(self, iterations=500):
        """Base teaches columns its representational space"""
        print("Creating fertile ground in columns...")

        for col_idx in range(4):
            if self.column_states[col_idx] != 'active':
                continue

            optimizer = torch.optim.AdamW(
                self.columns[col_idx].parameters(),
                lr=1e-4
            )

            for i in range(iterations):
                # Variable length sequences
                seq_len = torch.randint(10, 100, (1,)).item()
                input_ids = torch.randint(
                    0, self.vocab_size,
                    (1, seq_len)
                ).to(self.device)

                with torch.no_grad():
                    base_out = self.base(input_ids, output_hidden_states=True)
                    base_hidden = base_out.hidden_states[-1]

                # Column learns base's space with slight flexibility
                column_out = self.columns[col_idx](base_hidden)

                # Target is base + small noise for flexibility
                target = base_hidden + torch.randn_like(base_hidden) * 0.01

                loss = F.mse_loss(column_out, target)

                # Regularization to keep columns receptive
                loss += 0.001 * column_out.abs().mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print(f"Column {col_idx}: iteration {i}, loss: {loss.item():.4f}")

        print("Fertile ground prepared - columns ready for specialization")

    def check_token_hunger(self, input_text):
        """Determine if system needs more tokens before responding"""
        self.token_buffer.append(input_text)

        # Simple satisfaction metric based on buffer fullness and uncertainty
        recent_tokens = list(self.token_buffer)[-10:]
        if len(recent_tokens) < 3:
            self.current_satisfaction = 0.2
            return True  # Hungry for more

        # Check semantic completeness
        combined = " ".join(recent_tokens)
        if combined.count("?") > combined.count("."):
            self.current_satisfaction = 0.5
            return True  # Need more context

        self.current_satisfaction = 0.9
        return False  # Satisfied

    def generate_inquiry(self, uncertainty_type):
        """Generate questions to resolve uncertainty"""
        if uncertainty_type == 'high':
            return "I need more context about this. Can you elaborate?"
        elif uncertainty_type == 'mixed':
            return "This touches multiple areas. Which aspect should I focus on?"
        else:
            return "Tell me more so I can understand better."

    def live_inference(self, user_input):
        """Main inference with uncertainty-driven learning"""

        # Check token hunger
        if self.check_token_hunger(user_input):
            if self.current_satisfaction < self.satisfaction_threshold:
                return {
                    'mode': 'hungry',
                    'response': self.generate_inquiry('high'),
                    'satisfaction': self.current_satisfaction
                }

        # Tokenize input
        inputs = self.tokenizer(
            user_input,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)

        # Base model forward
        with torch.no_grad():
            base_out = self.base(**inputs, output_hidden_states=True)
            base_hidden = base_out.hidden_states[-1]
            base_logits = base_out.logits

        # Detect uncertainty
        uncertain_mask, prob_scores = self.detect_uncertainty(base_logits)

        if uncertain_mask.any():
            # Route through columns
            column_out = self.route_through_columns(
                base_hidden,
                uncertain_mask
            )

            # Learn from this uncertainty
            self.queue_uncertainty_pattern(
                inputs.input_ids,
                uncertain_mask,
                column_out
            )

            # Generate using column-enhanced representation
            response = self.generate_from_hidden(column_out)
        else:
            # Use base model for confident regions
            response = self.generate_from_hidden(base_hidden)

        return {
            'mode': 'responding',
            'response': response,
            'uncertainty_detected': uncertain_mask.any().item(),
            'satisfaction': self.current_satisfaction
        }

    def route_through_columns(self, hidden_states, uncertainty_mask):
        """Route uncertain patterns through appropriate columns"""

        # Find active columns
        active_cols = [
            i for i, state in enumerate(self.column_states)
            if state == 'active'
        ]

        if not active_cols:
            return hidden_states

        # For now, use first active column
        col_idx = active_cols[0]
        column_out = self.columns[col_idx](hidden_states)

        # Track this resolution attempt
        self.track_complementarity(col_idx, uncertainty_mask)

        # Combine with frozen columns if any
        if self.frozen_columns:
            frozen_outputs = []
            for frozen_idx in self.frozen_columns:
                frozen_out = self.columns[frozen_idx](hidden_states)
                frozen_outputs.append(frozen_out)

            # Average frozen column contributions
            frozen_combined = torch.stack(frozen_outputs).mean(dim=0)
            column_out = column_out + 0.3 * frozen_combined

        return column_out

    def track_complementarity(self, col_idx, uncertainty_resolved):
        """Track if column deserves freezing"""

        # Count successful uncertainty resolutions
        if uncertainty_resolved.sum() > 0:
            self.column_scores[col_idx] += uncertainty_resolved.float().mean().item()

        # Check for freezing threshold
        if self.column_scores[col_idx] > 50:  # Proven complementary
            self.freeze_column(col_idx)

    def freeze_column(self, col_idx):
        """Freeze column as part of extended core"""
        if self.column_states[col_idx] != 'frozen':
            print(f"Freezing column {col_idx} - proven complementary to base")

            self.column_states[col_idx] = 'frozen'
            self.frozen_columns.append(col_idx)

            # Freeze its parameters
            for param in self.columns[col_idx].parameters():
                param.requires_grad = False

            # Check if we need to activate new columns
            if len(self.frozen_columns) >= 2:
                self.activate_next_wave()

    def activate_next_wave(self):
        """Activate inactive columns when enough are frozen"""
        inactive = [
            i for i, state in enumerate(self.column_states)
            if state == 'inactive'
        ]

        if inactive:
            # For POC, just activate all inactive
            for idx in inactive[:4]:
                self.column_states[idx] = 'active'
                print(f"Activating column {idx} for new specialization")

    def check_homogeneity(self):
        """Check if frozen columns have become redundant"""
        if len(self.frozen_columns) < 2:
            return

        # Compare frozen column outputs on recent patterns
        recent_patterns = list(self.uncertainty_history)[-10:]
        if not recent_patterns:
            return

        similarities = []
        for pattern in recent_patterns:
            outputs = []
            for col_idx in self.frozen_columns:
                with torch.no_grad():
                    out = self.columns[col_idx](pattern.input_pattern)
                    outputs.append(out)

            # Check similarity between frozen columns
            if len(outputs) > 1:
                sim = F.cosine_similarity(outputs[0], outputs[1], dim=-1).mean()
                similarities.append(sim.item())

        # If too similar, unfreeze one
        if similarities and np.mean(similarities) > 0.95:
            unfroze_idx = self.frozen_columns.pop()
            self.column_states[unfroze_idx] = 'active'
            for param in self.columns[unfroze_idx].parameters():
                param.requires_grad = True
            print(f"Unfreezing column {unfroze_idx} due to homogeneity")

    def queue_uncertainty_pattern(self, input_ids, mask, response):
        """Queue patterns for background training"""
        event = UncertaintyEvent(
            timestamp=time.time(),
            input_pattern=input_ids,
            uncertainty_mask=mask,
            column_response=0,  # Track which column handled it
            resolved=False
        )
        self.uncertainty_history.append(event)
        self.training_queue.append(event)

    def background_training(self, iterations=10):
        """Process training queue during idle time"""
        if len(self.training_queue) < 10:
            return

        print(f"Background training with {len(self.training_queue)} patterns...")

        # Train active columns on queued patterns
        for col_idx, state in enumerate(self.column_states):
            if state != 'active':
                continue

            optimizer = torch.optim.AdamW(
                self.columns[col_idx].parameters(),
                lr=1e-5
            )

            for _ in range(min(iterations, len(self.training_queue))):
                event = self.training_queue.popleft()

                with torch.no_grad():
                    base_out = self.base(
                        event.input_pattern,
                        output_hidden_states=True
                    )
                    base_hidden = base_out.hidden_states[-1]

                column_out = self.columns[col_idx](base_hidden)

                # Learn to be different where uncertain
                uncertainty_weight = event.uncertainty_mask.float().unsqueeze(-1)
                loss = -torch.log(torch.sigmoid(
                    (column_out - base_hidden).abs() * uncertainty_weight
                )).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def generate_from_hidden(self, hidden_states):
        """Generate text from hidden states"""
        # Simple generation using last hidden state
        logits = self.output_projection(hidden_states[:, -1, :])

        # Sample token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs.squeeze(), 1)

        return self.tokenizer.decode(next_token)

    def run_continuous(self, duration_seconds=60):
        """24/7 operation simulation"""
        print(f"Running continuous operation for {duration_seconds} seconds...")

        start_time = time.time()
        cycle = 0

        while time.time() - start_time < duration_seconds:
            cycle += 1

            # Simulate user input (in reality, from STT or text)
            test_inputs = [
                "What is consciousness?",
                "Tell me about uncertainty",
                "How do neural networks learn?",
                "What is your name?",
                "Explain quantum computing"
            ]

            user_input = test_inputs[cycle % len(test_inputs)]

            # Process input
            result = self.live_inference(user_input)

            # Every 10 cycles, run background training
            if cycle % 10 == 0:
                self.background_training()
                self.check_homogeneity()

            # Small delay to simulate realistic timing
            time.sleep(0.5)

            if cycle % 20 == 0:
                self.print_status()

    def print_status(self):
        """Print current system status"""
        print("\n=== System Status ===")
        print(f"Column states: {self.column_states}")
        print(f"Frozen columns: {self.frozen_columns}")
        print(f"Column scores: {[f'{s:.2f}' for s in self.column_scores]}")
        print(f"Queue size: {len(self.training_queue)}")
        print(f"Satisfaction: {self.current_satisfaction:.2f}")
        print("=" * 20 + "\n")


def main():
    """Test the POC"""
    print("Initializing Uncertainty-Driven PNN...")
    model = UncertaintyPNN()

    # Phase 1: Create fertile ground
    print("\nPhase 1: Pretraining columns...")
    model.pretrain_fertile_ground(iterations=100)

    # Phase 2: Interactive testing
    print("\nPhase 2: Interactive inference...")

    test_queries = [
        "My name is",
        "Consciousness emerges from",
        "What is SEPAS?",
        "Tell me about progressive neural networks",
        "How does uncertainty relate to knowledge?"
    ]

    for query in test_queries:
        print(f"\nUser: {query}")
        result = model.live_inference(query)
        print(f"Mode: {result['mode']}")
        print(f"Response preview: {result['response'][:100]}...")
        print(f"Uncertainty detected: {result.get('uncertainty_detected', False)}")

    # Phase 3: Background training
    print("\nPhase 3: Processing background queue...")
    model.background_training(iterations=20)

    # Phase 4: Check system evolution
    model.print_status()

    # Optional: Run continuous operation
    # print("\nPhase 4: Continuous operation test...")
    # model.run_continuous(duration_seconds=30)


if __name__ == "__main__":
    main()