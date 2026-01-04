# Async Orchestration Implementation Flow

## Quick Start Guide

This document provides a step-by-step implementation path for building the Keko async task orchestration system. Start simple, test each phase, then add complexity.

## Phase 1: Basic Async Queue (Week 1)

### Step 1.1: Core Data Structures

```python
# keko/async_core.py

import asyncio
from enum import IntEnum, Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable
import time
import uuid

class TaskPriority(IntEnum):
    USER_WAITING = 0
    TOOL_RESEARCH = 1  
    COLUMN_QUERY = 2
    BACKGROUND = 3

@dataclass
class AsyncTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: TaskPriority = TaskPriority.BACKGROUND
    payload: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    result: Optional[Any] = None
    error: Optional[Exception] = None
    
    def __lt__(self, other):
        return self.priority < other.priority
```

### Step 1.2: Simple Queue Manager

```python
# keko/queue_manager.py

class SimpleQueueManager:
    def __init__(self):
        self.queue = asyncio.PriorityQueue()
        self.results = {}
        
    async def add_task(self, task: AsyncTask) -> str:
        """Add task to queue, return task_id"""
        await self.queue.put(task)
        return task.task_id
        
    async def process_tasks(self):
        """Simple worker loop"""
        while True:
            task = await self.queue.get()
            try:
                # Process based on priority
                if task.priority == TaskPriority.USER_WAITING:
                    result = await self.process_urgent(task)
                else:
                    result = await self.process_normal(task)
                    
                task.result = result
                self.results[task.task_id] = result
                
            except Exception as e:
                task.error = e
                self.results[task.task_id] = e
```

### Step 1.3: Test Basic Queue

```python
# tests/test_basic_queue.py

async def test_priority_ordering():
    manager = SimpleQueueManager()
    
    # Add tasks in wrong priority order
    await manager.add_task(AsyncTask(priority=TaskPriority.BACKGROUND))
    await manager.add_task(AsyncTask(priority=TaskPriority.USER_WAITING))
    await manager.add_task(AsyncTask(priority=TaskPriority.COLUMN_QUERY))
    
    # Verify USER_WAITING processed first
    # ... test implementation
```

### Milestone 1: Basic async queue working with priority ordering ✓

---

## Phase 2: Column Integration (Week 2)

### Step 2.1: Column Query Wrapper

```python
# keko/column_async.py

class AsyncColumnQuery:
    def __init__(self, columns, base_model):
        self.columns = columns
        self.base_model = base_model
        
    async def query_column(self, column_id: int, input_text: str):
        """Query a single column asynchronously"""
        # Wrap synchronous column forward pass
        return await asyncio.to_thread(
            self._sync_column_forward,
            column_id,
            input_text
        )
        
    def _sync_column_forward(self, column_id, input_text):
        """Synchronous forward pass (runs in thread)"""
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(input_text, return_tensors='pt')
            
            # Get hidden states from base
            hidden = self.base_model.get_hidden_states(inputs)
            
            # Pass through column
            if column_id == 0:
                output = hidden  # Base column
            else:
                output = self.columns[column_id](hidden)
                
            return output
```

### Step 2.2: Parallel Column Queries

```python
# keko/parallel_inference.py

class ParallelInference:
    async def query_all_columns(self, input_text: str, priority=TaskPriority.USER_WAITING):
        """Query all columns in parallel"""
        
        tasks = []
        for col_id in range(4):
            task = AsyncTask(
                priority=priority,
                payload={
                    "type": "column_query",
                    "column_id": col_id,
                    "input": input_text
                }
            )
            task_id = await self.queue_manager.add_task(task)
            tasks.append(task_id)
        
        # Wait for all with timeout
        results = await self.wait_for_tasks(tasks, timeout=2.0)
        
        # Calculate variance/agreement
        variance = self.calculate_variance(results)
        
        return {
            "responses": results,
            "variance": variance,
            "uncertainty": variance > self.threshold
        }
```

### Step 2.3: Test Parallel Columns

```python
# tests/test_parallel_columns.py

async def test_column_disagreement():
    inference = ParallelInference()
    
    # Test with ambiguous input
    result = await inference.query_all_columns(
        "The answer might be either 42 or 17"
    )
    
    assert result["variance"] > 0.5  # Expect disagreement
    assert result["uncertainty"] == True
```

### Milestone 2: Parallel column queries with variance measurement ✓

---

## Phase 3: Tool Integration (Week 3)

### Step 3.1: Tool Interface

```python
# keko/tools/base.py

from abc import ABC, abstractmethod

class BaseTool(ABC):
    @abstractmethod
    async def query(self, text: str) -> Dict:
        pass
    
    @abstractmethod
    def is_relevant(self, text: str) -> bool:
        pass

# keko/tools/wolfram.py

class WolframTool(BaseTool):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = aiohttp.ClientSession()
        
    async def query(self, text: str) -> Dict:
        """Query Wolfram Alpha API"""
        url = f"https://api.wolframalpha.com/v2/query"
        params = {
            "appid": self.api_key,
            "input": text,
            "format": "plaintext",
            "output": "json"
        }
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            return self.parse_wolfram_response(data)
    
    def is_relevant(self, text: str) -> bool:
        """Check if Wolfram would be helpful"""
        keywords = ["calculate", "math", "equation", "solve", "integral"]
        return any(k in text.lower() for k in keywords)
```

### Step 3.2: Tool Orchestration

```python
# keko/tool_orchestrator.py

class ToolOrchestrator:
    def __init__(self):
        self.tools = {
            "wolfram": WolframTool(api_key=os.getenv("WOLFRAM_KEY")),
            "search": SearchTool(),
            "calculator": LocalCalculator()
        }
    
    async def research_with_tools(self, query: str, uncertainty_score: float):
        """Research using relevant tools when uncertain"""
        
        if uncertainty_score < 0.5:
            return None  # Confident enough
        
        # Select relevant tools
        relevant = [
            name for name, tool in self.tools.items()
            if tool.is_relevant(query)
        ]
        
        if not relevant:
            return None
        
        # Queue tool tasks
        tool_tasks = []
        for tool_name in relevant:
            task = AsyncTask(
                priority=TaskPriority.TOOL_RESEARCH,
                payload={
                    "type": "tool_query",
                    "tool": tool_name,
                    "query": query
                }
            )
            tool_tasks.append(self.queue_manager.add_task(task))
        
        # Gather results
        results = await asyncio.gather(*tool_tasks)
        return self.aggregate_tool_results(results)
```

### Step 3.3: Test Tool Integration

```python
# tests/test_tools.py

async def test_wolfram_integration():
    orchestrator = ToolOrchestrator()
    
    result = await orchestrator.research_with_tools(
        "What is the integral of x^2 from 0 to 10?",
        uncertainty_score=0.8
    )
    
    assert "wolfram" in result
    assert result["wolfram"]["answer"] == "333.333..."
```

### Milestone 3: External tool queries working ✓

---

## Phase 4: Unified Pipeline (Week 4)

### Step 4.1: Main Inference Loop

```python
# keko/inference_pipeline.py

class InferencePipeline:
    def __init__(self):
        self.queue_manager = QueueManager()
        self.columns = AsyncColumnQuery()
        self.tools = ToolOrchestrator()
        self.verifier = VerificationPipeline()
        
    async def process_query(self, query: str, user_waiting=True):
        """Main inference pipeline"""
        
        priority = TaskPriority.USER_WAITING if user_waiting else TaskPriority.BACKGROUND
        
        # Step 1: Parallel column queries
        column_results = await self.columns.query_all_columns(query, priority)
        
        # Step 2: Check uncertainty
        if column_results["uncertainty"]:
            # Step 3: Tool research (parallel with step 4)
            tool_task = asyncio.create_task(
                self.tools.research_with_tools(query, column_results["variance"])
            )
            
            # Step 4: Request clarification (if needed)
            if column_results["variance"] > 0.8:
                clarification = await self.request_clarification(query)
                if clarification:
                    # Recursive call with clarified input
                    return await self.process_query(
                        query + " " + clarification,
                        user_waiting
                    )
            
            # Wait for tools
            tool_results = await tool_task
            
            # Step 5: Synthesize response
            response = self.synthesize_response(
                column_results,
                tool_results
            )
        else:
            # Confident response
            response = column_results["responses"][0]  # Base column
        
        # Step 6: Async verification (don't wait)
        asyncio.create_task(
            self.verifier.verify_async(response, query)
        )
        
        # Step 7: Queue for training (background)
        if column_results["uncertainty"]:
            await self.queue_training_sample(query, response)
        
        return response
```

### Step 4.2: Response Synthesis

```python
# keko/synthesis.py

class ResponseSynthesizer:
    def synthesize_response(self, column_results, tool_results=None):
        """Combine all sources into final response"""
        
        responses = column_results["responses"]
        
        # High agreement: weighted average
        if column_results["variance"] < 0.3:
            weights = self.calculate_weights(responses)
            return self.weighted_combine(responses, weights)
        
        # Disagreement but have tool results
        if tool_results:
            return self.tool_guided_synthesis(responses, tool_results)
        
        # Disagreement, no tools: acknowledge uncertainty
        return self.uncertain_response(responses)
    
    def uncertain_response(self, responses):
        """Generate response acknowledging uncertainty"""
        base_response = responses[0]
        
        return f"""I'm not entirely certain, but {base_response}
        
        Alternative perspectives from my analysis:
        {self.format_alternatives(responses[1:])}
        
        Would you like me to research this further or clarify anything?"""
```

### Milestone 4: Complete pipeline integration ✓

---

## Phase 5: Monitoring & Optimization (Week 5)

### Step 5.1: Metrics Collection

```python
# keko/monitoring.py

class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        
    async def record_task_completion(self, task: AsyncTask):
        """Record task metrics"""
        duration = time.time() - task.created_at
        
        self.metrics[f"{task.priority}_latency"].append(duration)
        
        if task.error:
            self.metrics["errors"].append(task.task_id)
        
        # Calculate percentiles
        if len(self.metrics[f"{task.priority}_latency"]) > 100:
            self.calculate_percentiles(task.priority)
    
    def get_dashboard_metrics(self):
        """Get current metrics for dashboard"""
        return {
            "user_p95_latency": self.get_percentile("USER_WAITING", 95),
            "avg_uncertainty": np.mean(self.metrics["uncertainty"]),
            "tool_usage": len(self.metrics["tool_queries"]),
            "error_rate": len(self.metrics["errors"]) / self.total_tasks
        }
```

### Step 5.2: Performance Optimization

```python
# keko/optimization.py

class PerformanceOptimizer:
    async def optimize_based_on_metrics(self, metrics):
        """Dynamically adjust system based on metrics"""
        
        # Scale workers if latency high
        if metrics["user_p95_latency"] > 5.0:
            await self.scale_workers(increase=2)
        
        # Adjust timeouts if many timeouts
        if metrics["timeout_rate"] > 0.1:
            self.increase_timeouts()
        
        # Cache frequent queries
        if metrics["query_frequency"].max() > 10:
            await self.enable_caching()
```

### Milestone 5: Monitoring and optimization working ✓

---

## Testing Strategy

### Integration Tests

```python
# tests/test_integration.py

class TestFullPipeline:
    async def test_uncertain_query_flow(self):
        """Test complete flow for uncertain query"""
        
        pipeline = InferencePipeline()
        
        # Ambiguous query that should trigger uncertainty
        response = await pipeline.process_query(
            "What happens when we divide by zero?"
        )
        
        # Verify uncertainty was detected
        assert pipeline.last_uncertainty_score > 0.5
        
        # Verify tools were called
        assert "wolfram" in pipeline.last_tool_results
        
        # Verify response acknowledges uncertainty
        assert "uncertain" in response.lower() or "might" in response.lower()
    
    async def test_confident_query_flow(self):
        """Test flow for confident query"""
        
        pipeline = InferencePipeline()
        
        response = await pipeline.process_query(
            "What is 2 + 2?"
        )
        
        # Should not trigger uncertainty
        assert pipeline.last_uncertainty_score < 0.3
        
        # Should not call tools
        assert pipeline.last_tool_results is None
        
        # Should give direct answer
        assert "4" in response
```

### Load Tests

```python
# tests/test_load.py

async def test_concurrent_users():
    """Test system under load"""
    
    pipeline = InferencePipeline()
    
    # Simulate 100 concurrent users
    queries = [
        pipeline.process_query(f"Query {i}", user_waiting=True)
        for i in range(100)
    ]
    
    start = time.time()
    results = await asyncio.gather(*queries)
    duration = time.time() - start
    
    # All should complete within reasonable time
    assert duration < 10.0  # 10 seconds for 100 queries
    
    # Check p95 latency
    latencies = [r.latency for r in results]
    p95 = np.percentile(latencies, 95)
    assert p95 < 2.0  # 2 second p95
```

---

## Deployment Guide

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers aiohttp asyncio numpy

# Set environment variables
export WOLFRAM_API_KEY="your-key"
export MAX_WORKERS=8
export GPU_DEVICE="cuda:0"
```

### Configuration File

```yaml
# keko/config.yaml

orchestration:
  max_workers: 8
  priorities:
    user_waiting_timeout: 5.0
    background_timeout: 60.0
  
columns:
  count: 4
  base_frozen: true
  
tools:
  wolfram:
    enabled: true
    timeout: 10.0
  search:
    enabled: false
    
monitoring:
  metrics_port: 9090
  dashboard_enabled: true
```

### Running the System

```python
# keko/main.py

async def main():
    # Load configuration
    config = load_config("config.yaml")
    
    # Initialize pipeline
    pipeline = InferencePipeline(config)
    
    # Start background workers
    workers = [
        asyncio.create_task(pipeline.queue_manager.process_tasks())
        for _ in range(config.max_workers)
    ]
    
    # Start metrics server
    asyncio.create_task(
        start_metrics_server(config.monitoring.metrics_port)
    )
    
    # Start API server
    app = create_fastapi_app(pipeline)
    await run_server(app, port=8000)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Debugging Guide

### Common Issues

1. **High Latency for User Queries**
   - Check worker count
   - Monitor GPU utilization
   - Review timeout settings

2. **Frequent Timeouts**
   - Increase timeout values
   - Check external API rate limits
   - Monitor network latency

3. **Memory Leaks**
   - Check task result cleanup
   - Monitor queue depths
   - Review cache sizes

### Debug Tools

```python
# keko/debug.py

class DebugInspector:
    def print_queue_state(self):
        """Print current queue state"""
        print(f"Queue depth: {self.queue.qsize()}")
        print(f"By priority:")
        for priority in TaskPriority:
            count = sum(1 for t in self.queue._queue if t.priority == priority)
            print(f"  {priority.name}: {count}")
    
    def print_active_tasks(self):
        """Print currently executing tasks"""
        for task_id, task in self.active_tasks.items():
            age = time.time() - task.created_at
            print(f"  {task_id}: {task.priority.name} (age: {age:.2f}s)")
```

---

## Next Steps

After completing all phases:

1. **Performance Tuning**
   - Profile bottlenecks
   - Optimize GPU batching
   - Implement result caching

2. **Advanced Features**
   - Task dependencies (DAG)
   - Distributed queue (Redis)
   - Circuit breakers
   - Predictive prefetching

3. **Production Hardening**
   - Add comprehensive logging
   - Implement health checks
   - Setup alerting
   - Add rate limiting

4. **Testing**
   - Chaos testing
   - Soak testing
   - Security testing
   - A/B testing different strategies

---

## Success Metrics

Your implementation is successful when:

- ✅ User queries complete in <2 seconds (p95)
- ✅ System handles 100+ concurrent requests
- ✅ Uncertain queries trigger tool research
- ✅ Column disagreement correlates with uncertainty
- ✅ Background training doesn't impact user latency
- ✅ Partial results available on timeout
- ✅ Monitoring dashboard shows real-time metrics

Start with Phase 1 and validate each milestone before moving forward. The system is designed to work at each phase, getting progressively more sophisticated.
