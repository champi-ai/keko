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


@dataclass
class ClarificationRecord:
    """Record of a clarification interaction"""
    timestamp: float
    original_input: str
    clarifying_question: str
    clarified_response: Optional[str]
    uncertainty_before: float
    uncertainty_after: Optional[float]
    clarification_gain: Optional[float]  # ΔU = U_before - U_after
    mode: str  # 'interactive' or 'autonomous'


class MetricsTracker:
    """Comprehensive metrics tracking for uncertainty-driven learning"""

    def __init__(self):
        # Token-level uncertainty tracking
        self.token_uncertainties = deque(maxlen=10000)  # Store individual token uncertainties
        self.uncertainty_timestamps = deque(maxlen=10000)  # Timestamps for temporal analysis

        # Uncertainty resolution tracking
        self.uncertainty_events = deque(maxlen=1000)  # Events with resolution status
        self.resolved_count = 0
        self.total_uncertain_events = 0

        # Temporal stability tracking
        self.epoch_uncertainties = deque(maxlen=100)  # Average uncertainty per epoch/cycle
        self.uncertainty_drift_history = deque(maxlen=50)  # TSM values over time

        # Performance metrics
        self.inference_times = deque(maxlen=500)
        self.generation_lengths = deque(maxlen=500)

        # Column performance
        self.column_activation_counts = [0, 0, 0, 0]
        self.column_success_rates = [0.0, 0.0, 0.0, 0.0]

    def record_token_uncertainty(self, uncertainty: float, timestamp: float = None):
        """Record per-token uncertainty"""
        if timestamp is None:
            timestamp = time.time()

        self.token_uncertainties.append(uncertainty)
        self.uncertainty_timestamps.append(timestamp)

    def record_uncertain_event(self, resolved: bool):
        """Record an uncertainty event and whether it was resolved"""
        self.total_uncertain_events += 1
        if resolved:
            self.resolved_count += 1

        self.uncertainty_events.append({
            'timestamp': time.time(),
            'resolved': resolved
        })

    def calculate_aut(self) -> float:
        """Calculate Average Uncertainty per Token (AUT)"""
        if not self.token_uncertainties:
            return 0.0

        return sum(self.token_uncertainties) / len(self.token_uncertainties)

    def calculate_urr(self) -> float:
        """Calculate Uncertainty Resolution Rate (URR)"""
        if self.total_uncertain_events == 0:
            return 0.0

        return self.resolved_count / self.total_uncertain_events

    def record_epoch_uncertainty(self):
        """Record average uncertainty for current epoch/cycle"""
        if not self.token_uncertainties:
            return

        # Get recent uncertainties (last 100 tokens)
        recent = list(self.token_uncertainties)[-100:]
        avg_uncertainty = sum(recent) / len(recent)

        self.epoch_uncertainties.append(avg_uncertainty)

        # Calculate TSM (Temporal Stability Metric) if we have history
        if len(self.epoch_uncertainties) >= 2:
            self._calculate_tsm()

    def _calculate_tsm(self):
        """Calculate Temporal Stability Metric - measures confidence drift"""
        if len(self.epoch_uncertainties) < 2:
            return 0.0

        # TSM = average absolute difference between consecutive epochs
        epochs = list(self.epoch_uncertainties)
        diffs = [abs(epochs[i] - epochs[i-1]) for i in range(1, len(epochs))]

        tsm = sum(diffs) / len(diffs)
        self.uncertainty_drift_history.append(tsm)

        return tsm

    def get_tsm(self) -> float:
        """Get latest Temporal Stability Metric"""
        if not self.uncertainty_drift_history:
            return 0.0

        return self.uncertainty_drift_history[-1]

    def record_column_activation(self, col_idx: int, success: bool):
        """Track column usage and success"""
        self.column_activation_counts[col_idx] += 1

        # Update success rate with exponential moving average
        alpha = 0.1  # Smoothing factor
        current_rate = self.column_success_rates[col_idx]
        self.column_success_rates[col_idx] = (
            alpha * (1.0 if success else 0.0) +
            (1 - alpha) * current_rate
        )

    def get_comprehensive_metrics(self) -> dict:
        """Get all metrics in one snapshot"""
        return {
            # Core uncertainty metrics
            'aut': self.calculate_aut(),
            'urr': self.calculate_urr(),
            'tsm': self.get_tsm(),

            # Resolution statistics
            'total_uncertain_events': self.total_uncertain_events,
            'resolved_events': self.resolved_count,
            'unresolved_events': self.total_uncertain_events - self.resolved_count,

            # Token statistics
            'total_tokens_tracked': len(self.token_uncertainties),
            'recent_avg_uncertainty': (
                sum(list(self.token_uncertainties)[-100:]) / min(100, len(self.token_uncertainties))
                if self.token_uncertainties else 0.0
            ),

            # Temporal analysis
            'epochs_recorded': len(self.epoch_uncertainties),
            'uncertainty_trend': self._get_uncertainty_trend(),
            'stability_trend': self._get_stability_trend(),

            # Column performance
            'column_activations': self.column_activation_counts,
            'column_success_rates': self.column_success_rates
        }

    def _get_uncertainty_trend(self) -> str:
        """Determine if uncertainty is increasing, decreasing, or stable"""
        if len(self.epoch_uncertainties) < 5:
            return "insufficient_data"

        recent = list(self.epoch_uncertainties)[-5:]
        first_half_avg = sum(recent[:3]) / 3
        second_half_avg = sum(recent[3:]) / 2

        diff = second_half_avg - first_half_avg

        if abs(diff) < 0.05:
            return "stable"
        elif diff > 0:
            return "increasing"
        else:
            return "decreasing"

    def _get_stability_trend(self) -> str:
        """Determine if model is becoming more or less stable"""
        if len(self.uncertainty_drift_history) < 5:
            return "insufficient_data"

        recent_drift = list(self.uncertainty_drift_history)[-5:]
        first_half_avg = sum(recent_drift[:3]) / 3
        second_half_avg = sum(recent_drift[3:]) / 2

        if abs(first_half_avg - second_half_avg) < 0.01:
            return "stable"
        elif second_half_avg < first_half_avg:
            return "stabilizing"  # Drift decreasing = more stable
        else:
            return "destabilizing"  # Drift increasing = less stable

    def print_metrics_summary(self):
        """Print human-readable metrics summary"""
        metrics = self.get_comprehensive_metrics()

        print("\n" + "="*60)
        print("📊 UNCERTAINTY METRICS SUMMARY")
        print("="*60)

        print(f"\n🎯 Core Metrics:")
        print(f"  AUT (Avg Uncertainty/Token): {metrics['aut']:.4f}")
        print(f"  URR (Resolution Rate):        {metrics['urr']:.2%}")
        print(f"  TSM (Temporal Stability):     {metrics['tsm']:.4f}")

        print(f"\n📈 Resolution Statistics:")
        print(f"  Total Uncertain Events: {metrics['total_uncertain_events']}")
        print(f"  Resolved:               {metrics['resolved_events']} ✓")
        print(f"  Unresolved:             {metrics['unresolved_events']}")

        print(f"\n📊 Trends:")
        print(f"  Uncertainty Trend: {metrics['uncertainty_trend']}")
        print(f"  Stability Trend:   {metrics['stability_trend']}")

        print(f"\n🔧 Column Performance:")
        for i, (count, rate) in enumerate(zip(metrics['column_activations'], metrics['column_success_rates'])):
            print(f"  Column {i}: {count:4d} activations, {rate:.1%} success rate")

        print("="*60 + "\n")


class ClarificationEngine:
    """Handles uncertainty resolution through clarification"""

    def __init__(self, tokenizer, base_model):
        self.tokenizer = tokenizer
        self.base_model = base_model

        # Clarification buffer - stores clarification history
        self.clarification_buffer = deque(maxlen=1000)

        # Dynamic threshold for triggering clarifications
        self.base_threshold = 0.6  # Base uncertainty threshold
        self.adaptive_threshold = 0.6
        self.threshold_momentum = 0.95  # For exponential moving average

        # Recent uncertainty statistics for adaptive thresholding
        self.recent_uncertainties = deque(maxlen=100)

        # Clarification templates
        self.clarification_templates = [
            "I'm not entirely certain about {topic}. Could you clarify what you mean?",
            "I have some uncertainty regarding {topic}. Can you provide more context?",
            "To better understand {topic}, could you elaborate on that?",
            "I need more information about {topic} to give you a confident answer.",
            "There are multiple interpretations of {topic}. Which aspect interests you most?",
        ]

    def dynamic_threshold(self):
        """Calculate adaptive threshold based on recent uncertainty patterns"""
        if len(self.recent_uncertainties) < 10:
            return self.base_threshold

        # Update threshold based on recent history
        mean_uncertainty = sum(self.recent_uncertainties) / len(self.recent_uncertainties)
        std_uncertainty = (
            sum((u - mean_uncertainty) ** 2 for u in self.recent_uncertainties) / len(self.recent_uncertainties)
        ) ** 0.5

        # Threshold = mean + 0.5 * std (captures elevated uncertainty)
        new_threshold = mean_uncertainty + 0.5 * std_uncertainty

        # Smooth with exponential moving average
        self.adaptive_threshold = (
            self.threshold_momentum * self.adaptive_threshold +
            (1 - self.threshold_momentum) * new_threshold
        )

        # Clamp between reasonable bounds
        return max(0.4, min(0.8, self.adaptive_threshold))

    def should_request_clarification(self, uncertainty_score: float, mode: str = 'interactive') -> bool:
        """Determine if clarification should be requested"""
        # Track uncertainty for adaptive threshold
        self.recent_uncertainties.append(uncertainty_score)

        threshold = self.dynamic_threshold()

        # In interactive mode, lower threshold (more willing to ask)
        if mode == 'interactive':
            threshold *= 0.9

        return uncertainty_score > threshold

    def generate_clarification_question(self, input_text: str, uncertainty_score: float) -> str:
        """Generate a clarifying question based on input and uncertainty"""
        # Extract potential topic (simple heuristic: last noun phrase or keywords)
        words = input_text.split()

        # Find key content words (simple approach)
        topic = "this"
        if len(words) > 3:
            # Take last few words as topic
            topic = " ".join(words[-3:])
        elif words:
            topic = " ".join(words)

        # Select template based on uncertainty level
        if uncertainty_score > 0.8:
            template_idx = 0  # Most uncertain
        elif uncertainty_score > 0.7:
            template_idx = 1
        else:
            template_idx = 2

        template = self.clarification_templates[template_idx % len(self.clarification_templates)]

        return template.format(topic=topic)

    def request_clarification(
        self,
        input_text: str,
        uncertainty_score: float,
        mode: str = 'interactive'
    ) -> Optional[str]:
        """Main entry point: request clarification if needed"""

        if not self.should_request_clarification(uncertainty_score, mode):
            return None

        question = self.generate_clarification_question(input_text, uncertainty_score)

        # Record the clarification request
        record = ClarificationRecord(
            timestamp=time.time(),
            original_input=input_text,
            clarifying_question=question,
            clarified_response=None,
            uncertainty_before=uncertainty_score,
            uncertainty_after=None,
            clarification_gain=None,
            mode=mode
        )

        self.clarification_buffer.append(record)

        return question

    def record_clarification_result(
        self,
        original_input: str,
        clarified_response: str,
        uncertainty_after: float
    ):
        """Update clarification record with results and calculate gain"""
        # Find the most recent matching clarification request
        for record in reversed(self.clarification_buffer):
            if record.original_input == original_input and record.uncertainty_after is None:
                record.clarified_response = clarified_response
                record.uncertainty_after = uncertainty_after
                record.clarification_gain = record.uncertainty_before - uncertainty_after
                break

    def get_clarification_efficiency(self) -> float:
        """Calculate average clarification gain (ΔU) across recent clarifications"""
        gains = [
            r.clarification_gain
            for r in self.clarification_buffer
            if r.clarification_gain is not None
        ]

        if not gains:
            return 0.0

        return sum(gains) / len(gains)

    def get_clarification_stats(self) -> dict:
        """Get statistics about clarification effectiveness"""
        total_clarifications = len(self.clarification_buffer)
        completed = sum(1 for r in self.clarification_buffer if r.clarified_response is not None)
        successful = sum(1 for r in self.clarification_buffer if r.clarification_gain and r.clarification_gain > 0.1)

        return {
            'total_requests': total_clarifications,
            'completed': completed,
            'successful_resolutions': successful,
            'avg_clarification_gain': self.get_clarification_efficiency(),
            'success_rate': successful / completed if completed > 0 else 0.0,
            'adaptive_threshold': self.adaptive_threshold
        }


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

        # Generation uncertainty tracking
        self.generation_uncertainty = deque(maxlen=500)  # Track last 500 generations

        # Clarification engine
        self.clarification_engine = ClarificationEngine(self.tokenizer, self.base)

        # Metrics tracking
        self.metrics = MetricsTracker()

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
        """Determine if system needs more tokens before responding with enhanced heuristics"""
        self.token_buffer.append(input_text)

        # Get recent context
        recent_tokens = list(self.token_buffer)[-10:]
        combined = " ".join(recent_tokens)

        # Multi-factor satisfaction assessment
        satisfaction_factors = []

        # Factor 1: Buffer fullness (need minimum context)
        if len(recent_tokens) < 3:
            satisfaction_factors.append(0.2)
        else:
            satisfaction_factors.append(0.8)

        # Factor 2: Token count (prefer substantial inputs)
        token_count = len(input_text.split())
        if token_count < 5:
            satisfaction_factors.append(0.3)
        elif token_count < 15:
            satisfaction_factors.append(0.6)
        else:
            satisfaction_factors.append(1.0)

        # Factor 3: Semantic completeness markers
        question_marks = combined.count("?")
        periods = combined.count(".")
        if question_marks > periods:
            # More questions than statements - might need clarification
            satisfaction_factors.append(0.5)
        elif periods == 0 and len(combined) > 20:
            # Long input with no clear end - might be incomplete
            satisfaction_factors.append(0.6)
        else:
            satisfaction_factors.append(0.9)

        # Factor 4: Information density (check for vague/incomplete inputs)
        vague_indicators = ['um', 'uh', 'well', 'maybe', 'sort of', 'kind of', '...']
        vague_count = sum(1 for indicator in vague_indicators if indicator in input_text.lower())
        if vague_count > 2:
            satisfaction_factors.append(0.4)
        else:
            satisfaction_factors.append(0.8)

        # Factor 5: Referential completeness (avoid pronouns without context)
        pronouns = ['it', 'this', 'that', 'these', 'those', 'they', 'them']
        first_words = input_text.lower().split()[:3]
        if any(pronoun in first_words for pronoun in pronouns) and len(recent_tokens) < 2:
            # Pronoun reference without sufficient context
            satisfaction_factors.append(0.3)
        else:
            satisfaction_factors.append(0.9)

        # Calculate weighted average satisfaction
        self.current_satisfaction = sum(satisfaction_factors) / len(satisfaction_factors)

        # Hungry if satisfaction is below threshold
        return self.current_satisfaction < self.satisfaction_threshold

    def generate_inquiry(self, uncertainty_type, input_text=None):
        """Generate contextual questions to resolve uncertainty"""
        if input_text:
            # Analyze input to generate contextual inquiry
            token_count = len(input_text.split())

            if token_count < 5:
                return "Could you provide more details? I need more context to understand your request."
            elif input_text.lower().startswith(('it', 'this', 'that', 'these', 'those')):
                return "What are you referring to? I need more context about what you're asking about."
            elif '?' not in input_text and '.' not in input_text:
                return "I notice your message seems incomplete. Could you finish your thought?"
            else:
                return "Could you elaborate on that? I want to make sure I understand correctly."

        # Fallback to type-based inquiries
        if uncertainty_type == 'high':
            return "I need more context about this. Can you elaborate?"
        elif uncertainty_type == 'mixed':
            return "This touches multiple areas. Which aspect should I focus on?"
        else:
            return "Tell me more so I can understand better."

    def process_clarified_input(self, original_input: str, clarified_input: str):
        """Process user's clarified response and track improvement"""
        # Run inference on clarified input
        result = self.live_inference(clarified_input)

        # Record the clarification result with uncertainty after
        if 'uncertainty_score' in result:
            self.clarification_engine.record_clarification_result(
                original_input=original_input,
                clarified_response=clarified_input,
                uncertainty_after=result['uncertainty_score']
            )

        return result

    def live_inference(self, user_input, is_clarified=False):
        """Main inference with uncertainty-driven learning

        Args:
            user_input: The input text to process
            is_clarified: If True, skip clarification (this is already a clarified input)
        """

        # Check token hunger (unless this is a clarified response)
        if not is_clarified and self.check_token_hunger(user_input):
            if self.current_satisfaction < self.satisfaction_threshold:
                return {
                    'mode': 'hungry',
                    'response': self.generate_inquiry('high', user_input),
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

        # Calculate average uncertainty score for this input
        avg_uncertainty = (1.0 - prob_scores.mean().item())

        if uncertain_mask.any():
            # Record uncertain event (will update resolution status later)
            self.metrics.record_uncertain_event(resolved=False)  # Assume unresolved initially

            # Check if we should request clarification (only if not already clarified)
            if not is_clarified:
                clarification_question = self.clarification_engine.request_clarification(
                    user_input,
                    avg_uncertainty,
                    mode='interactive'
                )

                if clarification_question:
                    return {
                        'mode': 'clarifying',
                        'response': clarification_question,
                        'uncertainty_score': avg_uncertainty,
                        'requires_clarification': True,
                        'satisfaction': self.current_satisfaction
                    }

            # Route through columns (high uncertainty but no clarification needed)
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

            # Check if uncertainty was reduced after column processing (simple heuristic)
            # In a full implementation, we'd re-evaluate uncertainty after generation
            if is_clarified:
                # If this was a clarified input, mark previous event as resolved
                self.metrics.resolved_count += 1
        else:
            # Use base model for confident regions
            response = self.generate_from_hidden(base_hidden)

        return {
            'mode': 'responding',
            'response': response,
            'uncertainty_detected': uncertain_mask.any().item(),
            'uncertainty_score': avg_uncertainty,
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

    def generate_from_hidden(self, hidden_states, max_new_tokens=50, temperature=0.7, top_k=50):
        """Generate text autoregressively from hidden states with uncertainty tracking"""
        # Get initial logits from last hidden state
        current_hidden = hidden_states[:, -1:, :]  # Keep batch and sequence dims

        generated_tokens = []
        uncertainty_per_token = []

        for step in range(max_new_tokens):
            # Project to vocabulary
            logits = self.output_projection(current_hidden.squeeze(1))

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs.squeeze(), 1)

            # Track uncertainty for this token
            max_prob = probs.max().item()
            uncertainty = 1.0 - max_prob
            uncertainty_per_token.append(uncertainty)

            # Record in metrics tracker
            self.metrics.record_token_uncertainty(uncertainty)

            # Check stopping conditions
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Decode to check for natural stopping
            decoded = self.tokenizer.decode(next_token)
            if decoded in ['.', '!', '?', '\n'] and len(generated_tokens) > 10:
                generated_tokens.append(next_token.item())
                break

            generated_tokens.append(next_token.item())

            # Get next hidden state by running through base model
            # For efficiency, we could cache past key-values in future
            with torch.no_grad():
                try:
                    # Try to get embeddings for next token
                    token_embed = self.base.get_input_embeddings()(next_token.unsqueeze(0))

                    # Get next hidden state (simplified - full implementation would use past_key_values)
                    next_hidden_output = self.base(
                        inputs_embeds=token_embed,
                        output_hidden_states=True,
                        use_cache=False
                    )
                    current_hidden = next_hidden_output.hidden_states[-1]

                except Exception as e:
                    # Fallback: reuse last hidden state (less accurate but more robust)
                    print(f"Warning: Could not get next hidden state, reusing last: {e}")
                    break

        # Decode full sequence
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Calculate average uncertainty for generated sequence
        avg_uncertainty = sum(uncertainty_per_token) / len(uncertainty_per_token) if uncertainty_per_token else 0.0

        # Store uncertainty history for this generation
        if hasattr(self, 'generation_uncertainty'):
            self.generation_uncertainty.append({
                'response': response,
                'avg_uncertainty': avg_uncertainty,
                'uncertainty_per_token': uncertainty_per_token,
                'num_tokens': len(generated_tokens)
            })

        return response

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

                # Record epoch uncertainty for TSM calculation
                self.metrics.record_epoch_uncertainty()

            # Small delay to simulate realistic timing
            time.sleep(0.5)

            if cycle % 20 == 0:
                self.print_status()

    def print_status(self):
        """Print comprehensive system status with metrics"""
        print("\n" + "="*60)
        print("🧠 KEKO UNCERTAINTY PNN STATUS")
        print("="*60)

        # Column status
        print(f"\n📊 Column Status:")
        print(f"  States:  {self.column_states}")
        print(f"  Frozen:  {self.frozen_columns}")
        print(f"  Scores:  {[f'{s:.2f}' for s in self.column_scores]}")

        # System health
        print(f"\n💚 System Health:")
        print(f"  Training Queue: {len(self.training_queue):4d} samples")
        print(f"  Token Satisfaction: {self.current_satisfaction:.2%}")
        print(f"  Token Buffer: {len(self.token_buffer):4d} recent inputs")

        # Clarification statistics
        clarification_stats = self.clarification_engine.get_clarification_stats()
        print(f"\n💬 Clarification Engine:")
        print(f"  Total Requests:    {clarification_stats['total_requests']:4d}")
        print(f"  Completed:         {clarification_stats['completed']:4d}")
        print(f"  Successful:        {clarification_stats['successful_resolutions']:4d}")
        print(f"  Success Rate:      {clarification_stats['success_rate']:.1%}")
        print(f"  Avg Gain (ΔU):     {clarification_stats['avg_clarification_gain']:.4f}")
        print(f"  Current Threshold: {clarification_stats['adaptive_threshold']:.3f}")

        # Core uncertainty metrics
        print(f"\n📈 Uncertainty Metrics:")
        print(f"  AUT (Avg Uncertainty): {self.metrics.calculate_aut():.4f}")
        print(f"  URR (Resolution Rate): {self.metrics.calculate_urr():.2%}")
        print(f"  TSM (Temporal Stability): {self.metrics.get_tsm():.4f}")

        # Trends
        metrics = self.metrics.get_comprehensive_metrics()
        print(f"\n📉 Trends:")
        print(f"  Uncertainty:  {metrics['uncertainty_trend']}")
        print(f"  Stability:    {metrics['stability_trend']}")

        print("="*60 + "\n")


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