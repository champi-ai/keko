import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pickle
import time
from collections import deque


@dataclass
class MemoryEntry:
    embedding: np.ndarray
    text: str
    timestamp: float
    task_column: int = 0  # Which PNN column this belongs to
    access_count: int = 0
    importance: float = 1.0


class ProgressiveNeuralNetwork(nn.Module):
    """Progressive Neural Network for expandable learning without forgetting"""

    def __init__(self, hidden_size=960, base_model=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.base_model = base_model

        # Column management
        self.columns = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.column_names = []

        # Add base column (frozen)
        self.add_base_column()

    def add_base_column(self):
        """Add the frozen base model as column 0"""
        self.columns.append(nn.Identity())  # Placeholder for base model
        self.column_names.append("base")

    def add_column(self, name="new_task", layers=3):
        """Add a new progressive column"""
        col_idx = len(self.columns)

        # Create new column layers
        column_layers = nn.ModuleList()
        for i in range(layers):
            column_layers.append(nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1)
            ))

        # Create lateral connections from all previous columns
        if col_idx > 0:
            lateral_layers = nn.ModuleList()
            for layer_idx in range(layers):
                # Connection from all previous columns to this layer
                lateral_layers.append(
                    nn.Linear(self.hidden_size * col_idx, self.hidden_size)
                )
            self.lateral_connections.append(lateral_layers)

            # Adapter to combine this column with previous ones
            self.adapters.append(
                nn.Linear(self.hidden_size * (col_idx + 1), self.hidden_size)
            )

        self.columns.append(column_layers)
        self.column_names.append(name)
        # Move new components to CUDA if available
        if torch.cuda.is_available():
            for layer in column_layers:
                layer.cuda()
            if col_idx > 0:
                for layer in self.lateral_connections[-1]:
                    layer.cuda()
                self.adapters[-1].cuda()

        print(f"Added column {col_idx}: {name}")
        return col_idx

    def forward(self, hidden_states, active_columns=None):
        """Forward pass through specified columns
        Args:
            hidden_states: Already computed hidden states from base model
        """
        if active_columns is None:
            active_columns = list(range(len(self.columns)))

        # Store outputs from each column
        column_outputs = []

        for col_idx in active_columns:
            if col_idx == 0:
                # Base column - just pass through the hidden states
                col_output = hidden_states
            else:
                # Progressive column processing
                col_input = hidden_states

                for layer_idx, layer in enumerate(self.columns[col_idx]):
                    # Get lateral inputs if available
                    if col_idx > 1 and (col_idx - 1) < len(self.lateral_connections):
                        if layer_idx < len(self.lateral_connections[col_idx - 1]):
                            # Simple lateral connection from previous output
                            lateral_input = torch.cat([c for c in column_outputs], dim=-1)
                            lateral_output = self.lateral_connections[col_idx - 1][layer_idx](lateral_input)
                            col_input = layer(col_input) + 0.1 * lateral_output
                        else:
                            col_input = layer(col_input)
                    else:
                        col_input = layer(col_input)

                col_output = col_input

            column_outputs.append(col_output)

        # Combine outputs
        if len(column_outputs) > 1 and len(self.adapters) > 0:
            combined = torch.cat(column_outputs, dim=-1)
            final_output = self.adapters[-1](combined)
        else:
            final_output = column_outputs[-1] if column_outputs else hidden_states

        return final_output


class EWC:
    """Elastic Weight Consolidation for continual learning"""

    def __init__(self, model, importance=1000):
        self.model = model
        self.importance = importance
        self.params = {}
        self.fisher = {}

    def consolidate(self, dataset):
        """Compute Fisher information matrix after task"""
        # Store current parameters
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.clone().detach()
                self.fisher[n] = torch.zeros_like(p)

        # Compute Fisher information
        self.model.eval()
        for batch_idx, batch in enumerate(dataset):
            self.model.zero_grad()
            output = self.model(batch)
            # Use negative log likelihood or other loss
            loss = output.sum()  # Simplified
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.detach() ** 2

        # Average Fisher information
        for n in self.fisher:
            self.fisher[n] /= len(dataset)

        self.model.train()

    def penalty(self):
        """Compute EWC penalty for current parameters"""
        loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.importance * loss


class ExpandableModelPNN(nn.Module):
    """Full expandable model with PNN and memory"""

    def __init__(self, base_model_name="HuggingFaceTB/SmolLM-360M-Instruct"):
        # ... existing init code ...
        super().__init__()

        # Load models
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # Add this for better generat
        # Freeze base
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.hidden_size = self.base_model.config.hidden_size
        vocab_size = self.base_model.config.vocab_size

        # Initialize PNN
        self.pnn = ProgressiveNeuralNetwork(self.hidden_size)

        # Output projection for PNN (initialized from base model)
        self.output_projection = nn.Linear(self.hidden_size, vocab_size)
        if hasattr(self.base_model, 'lm_head'):
            with torch.no_grad():
                self.output_projection.weight.copy_(self.base_model.lm_head.weight)
                if self.base_model.lm_head.bias is not None:
                    self.output_projection.bias.copy_(self.base_model.lm_head.bias)

        # Memory and tracking
        self.memory_bank = MemoryBank(self.hidden_size)
        self.interaction_count = 0
        self.current_column = 0
        self.task_memories = {}
        # Move everything to CUDA at the end of init
        if torch.cuda.is_available():
            self.base_model = self.base_model.cuda()
            self.pnn = self.pnn.cuda()
            self.output_projection = self.output_projection.cuda()

    def generate_ensemble(self, prompt, max_length=100000, pnn_weight=0.3):
        """Generate by combining base model and PNN outputs"""
        messages = [{"role": "user", "content": prompt}]

        # # Add memory context
        # relevant_memories = self.memory_bank.search_memories(prompt, k=3)
        # if relevant_memories:
        #     memory_text = "\n".join([m.text for m in relevant_memories])
        #     messages.insert(0, {
        #         "role": "system",
        #         "content": f"Previous context:\n{memory_text}"
        #     })

        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        )
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        generated = inputs

        for _ in range(max_length):
            with torch.no_grad():
                # 1. Get base model predictions
                base_outputs = self.base_model(generated, output_hidden_states=True)
                base_logits = base_outputs.logits[:, -1, :]

                # 2. Get PNN-enhanced predictions
                hidden = base_outputs.hidden_states[-1]
                enhanced_hidden = self.pnn(hidden, list(range(len(self.pnn.columns))))

                # Project PNN output to vocabulary
                # (Need to add this layer in __init__)
                pnn_logits = self.output_projection(enhanced_hidden[:, -1, :])

                # 3. Combine logits (weighted average in probability space)
                base_probs = F.softmax(base_logits, dim=-1)
                pnn_probs = F.softmax(pnn_logits, dim=-1)

                # Weighted combination
                combined_probs = (1 - pnn_weight) * base_probs + pnn_weight * pnn_probs

                # 4. Sample from combined distribution
                next_token = torch.multinomial(combined_probs, 1)
                generated = torch.cat([generated, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        response = self.tokenizer.decode(
            generated[0][len(inputs[0]):],
            skip_special_tokens=True
        )
        return response

    def add_task(self, task_name):
        """Add a new task/knowledge domain"""
        col_idx = self.pnn.add_column(task_name)
        self.task_memories[task_name] = []
        self.pnn = self.pnn.cuda()
        return col_idx

    def forward(self, input_ids, column_idx=None):
        """Forward pass through expandable model"""
        # Get base model embeddings
        with torch.no_grad():
            outputs = self.base_model(
                input_ids,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]

        # Ensure hidden states are float
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()

        # Process through PNN
        if column_idx is None:
            column_idx = self.current_column

        active_columns = list(range(min(column_idx + 1, len(self.pnn.columns))))
        enhanced_states = self.pnn(hidden_states, active_columns)

        if torch.cuda.is_available() and not enhanced_states.is_cuda:
            enhanced_states = enhanced_states.cuda()

        # Integrate memory
        memory_context = self.memory_bank.retrieve(enhanced_states)
        if memory_context is not None:
            if not memory_context.is_cuda and torch.cuda.is_available():
                memory_context = memory_context.cuda()
            enhanced_states = enhanced_states + 0.1 * memory_context.unsqueeze(0).unsqueeze(0)

        return enhanced_states

    def learn_from_interaction(self, text, response, task_name=None):
        """Learn from a single interaction"""
        if task_name:
            if task_name not in self.task_memories:
                # New task - create column
                self.current_column = self.add_task(task_name)
                self.task_memories[task_name] = []
            else:
                # Existing task - find its column
                if task_name in self.pnn.column_names:
                    self.current_column = self.pnn.column_names.index(task_name)

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1000)
        targets = self.tokenizer(response, return_tensors="pt", truncation=True, max_length=1000)

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            targets = {k: v.cuda() for k, v in targets.items()}

        # Get hidden states
        with torch.no_grad():
            hidden = self.forward(inputs['input_ids'])

        # Store in memory
        self.memory_bank.store(
            hidden.mean(dim=1),
            text,
            task_column=self.current_column
        )

        # Simple learning step (if progressive column exists)
        if self.current_column > 0:
            self.pnn = self.pnn.cuda()
            optimizer = torch.optim.AdamW(
                [p for p in self.pnn.columns[self.current_column].parameters()],
                lr=1e-4
            )

            # Forward pass
            output = self.forward(inputs['input_ids'])

            # Simple loss (you'd want something better in practice)
            loss = F.mse_loss(output.mean(), torch.randn_like(output.mean()))

            # Add EWC penalty if it exists
            # if self.ewc is not None:
            #     loss += self.ewc.penalty()

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.interaction_count += 1

    def generate_with_memory(self, prompt, max_length=10000):
        """Generate text using full expandable system"""
        messages = [{"role": "user", "content": prompt}]

        # Retrieve relevant memories
        relevant_memories = self.memory_bank.search_memories(prompt, k=3)

        if relevant_memories:
            # Add memory context
            memory_text = "\n".join([m.text for m in relevant_memories])
            messages.insert(0, {
                "role": "system",
                "content": f"Previous context:\n{memory_text}"
            })

        # Tokenize with memory context
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            pad=True
        )

        # Create attention mask
        attention_mask = torch.ones_like(inputs)

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            attention_mask = attention_mask.cuda()

        # Generate
        with torch.no_grad():
            outputs = self.base_model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return response

    def save_state(self, path="expandable_pnn_state.pt"):
        """Save complete model state"""
        state = {
            'pnn_state': self.pnn.state_dict(),
            'memories': self.memory_bank.memories,
            'interaction_count': self.interaction_count,
            'current_column': self.current_column,
            'column_names': self.pnn.column_names,
            'task_memories': self.task_memories
        }
        torch.save(state, path)
        print(f"Saved state to {path}")

    def load_state(self, path="expandable_pnn_state.pt"):
        """Load complete model state"""
        if os.path.exists(path):
            try:
                state = torch.load(path, weights_only=False)  # Add weights_only=False

                # Rebuild PNN structure to match saved state
                self.pnn = ProgressiveNeuralNetwork(self.hidden_size)

                # Recreate columns to match saved structure
                for col_name in state['column_names'][1:]:  # Skip 'base'
                    self.pnn.add_column(col_name)

                # Now load the state dict
                self.pnn.load_state_dict(state['pnn_state'])

                # Restore other attributes
                self.memory_bank.memories = state['memories']
                self.interaction_count = state['interaction_count']
                self.current_column = state['current_column']
                self.task_memories = state['task_memories']

                if torch.cuda.is_available():
                    self.pnn = self.pnn.cuda()

                print(f"Loaded state from {path}")
                print(f"Restored {len(self.pnn.columns)} columns, {len(self.memory_bank.memories)} memories")
            except Exception as e:
                print(f"Failed to load state: {e}")
                print("Starting fresh")


class MemoryBank:
    """Enhanced memory bank with search capabilities"""

    def __init__(self, embedding_dim, max_size=100000):
        self.embedding_dim = embedding_dim
        self.max_size = max_size
        self.memories = deque(maxlen=max_size)

    def store(self, embedding, text, task_column=0):
        """Store memory with task association"""
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        memory = MemoryEntry(
            embedding=embedding.flatten(),
            text=text,
            timestamp=time.time(),
            task_column=task_column
        )
        self.memories.append(memory)

    def search_memories(self, query_text, k=5):
        """Search memories by text similarity"""
        if not self.memories:
            return []

        # Simple keyword matching (you'd want better in practice)
        query_words = set(query_text.lower().split())
        scored_memories = []

        for memory in self.memories:
            memory_words = set(memory.text.lower().split())
            score = len(query_words & memory_words) / max(len(query_words), 1)
            if score > 0:
                scored_memories.append((score, memory))

        # Sort by score and return top k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored_memories[:k]]

    def retrieve(self, query_embedding, k=5):
        """Retrieve by embedding similarity"""
        if not self.memories or query_embedding is None:
            return None

        if isinstance(query_embedding, torch.Tensor):
            query = query_embedding.mean(dim=1).detach().cpu().numpy().flatten()
        else:
            query = query_embedding.flatten()

        similarities = []
        for memory in list(self.memories)[-1000:]:  # Check recent memories
            sim = np.dot(query, memory.embedding) / (
                    np.linalg.norm(query) * np.linalg.norm(memory.embedding) + 1e-8
            )
            similarities.append((sim, memory))

        # Get top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_memories = [m for _, m in similarities[:k]]

        if top_memories:
            avg_embedding = np.mean([m.embedding for m in top_memories], axis=0)
            device = query_embedding.device if isinstance(query_embedding, torch.Tensor) else (
                'cuda' if torch.cuda.is_available() else 'cpu')
            return torch.tensor(avg_embedding, dtype=torch.float32, device=device)
        return None


# Test functions
def test_progressive_learning(model):
    """Test the progressive learning system"""
    import os


    # Learning sequence
    interactions = [
        ("My name is Vrykolakas", "personal_info"),
        ("I'm working on expandable AI architectures", "personal_info"),
        ("SEPAS means Self-Emergent Presence Acknowledgement System", "concepts"),
        ("Consciousness emerges from uncertainty in my theory", "concepts"),
        ("Progressive Neural Networks prevent catastrophic forgetting", "technical"),
    ]

    print("\n=== Progressive Learning Test ===\n")

    for text, task in interactions:
        print(f"Learning: {text} (Task: {task})")

        # Generate response before learning
        response = model.generate_ensemble(text, pnn_weight=0)

        print(f"Response: {response}...")

        # Learn from interaction
        model.learn_from_interaction(text, response, task)
        print(f"Learned! Column: {model.current_column}, Memories: {len(model.memory_bank.memories)}\n")

    # Test memory recall
    test_queries = [
        "What's my name?",
        "What am I working on?",
        "What is SEPAS?",
        "Tell me about consciousness and uncertainty"
    ]

    print("\n=== Memory Recall Test ===\n")

    for query in test_queries:
        print(f"Query: {query}")
        response = model.generate_ensemble(query, pnn_weight=0.02)
        print(f"Response: {response}...")
        print("-" * 50)

    # Save state
    model.save_state()
    print("\nState saved successfully!")


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    for i in range(0, 1):
        model = ExpandableModelPNN()
        if device == 'cuda':
            model = model.cuda()
        # model.load_state()  # Load previous learning each time
        test_progressive_learning(model=model)


if __name__ == "__main__":
    main()