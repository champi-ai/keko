# Async Task Orchestration System Specification

## Overview

The Keko Async Task Orchestration System manages all inference, research, and training operations through a priority-based asynchronous queue. This ensures user-facing operations receive immediate resources while background tasks utilize idle capacity efficiently.

## Core Architecture

### Task Priority Levels

```python
class TaskPriority(IntEnum):
    USER_WAITING = 0      # Highest - user actively waiting for response
    CLARIFICATION = 1     # User interaction needed (clarification requests)
    TOOL_RESEARCH = 2     # External API calls (Wolfram, search engines)
    COLUMN_QUERY = 3      # Internal model column queries
    VERIFICATION = 4      # Response verification pipeline
    BACKGROUND_TRAIN = 5  # Passive adaptation and training
```

### Task Types

```python
class TaskType(Enum):
    COLUMN_INFERENCE = "column"      # Query model columns
    TOOL_CALL = "tool"               # External tool/API calls
    USER_REQUEST = "user"            # Direct user interactions
    VERIFICATION = "verify"          # Verification encoder checks
    TRAINING = "train"              # Background training tasks
    CLARIFICATION = "clarify"        # Clarification generation
```

## Task Structure

```python
@dataclass
class AsyncTask:
    # Core properties
    task_id: str                    # Unique identifier
    priority: TaskPriority           # Execution priority
    task_type: TaskType             # Type of operation
    payload: Dict                   # Task-specific data
    
    # Execution control
    timeout: float = 30.0           # Maximum execution time
    retries: int = 3                # Retry attempts on failure
    callback: Optional[Callable]    # Completion callback
    
    # Metadata
    created_at: float               # Creation timestamp
    started_at: Optional[float]     # Execution start time
    completed_at: Optional[float]   # Completion time
    user_session_id: Optional[str]  # Associated user session
    
    # Results
    result: Optional[Any] = None
    error: Optional[Exception] = None
    
    def __lt__(self, other):
        """Priority queue ordering"""
        if self.priority != other.priority:
            return self.priority < other.priority
        # Same priority: older tasks first (FIFO within priority)
        return self.created_at < other.created_at
```

## Resource Management

### Semaphore Configuration

```python
class ResourcePool:
    # GPU resources for model inference
    gpu_inference_semaphore = asyncio.Semaphore(4)     # Max parallel GPU ops
    
    # External API rate limiting
    wolfram_semaphore = asyncio.Semaphore(5)           # Wolfram API limit
    search_semaphore = asyncio.Semaphore(10)           # Search API limit
    
    # CPU-bound operations
    verification_semaphore = asyncio.Semaphore(8)      # Verification threads
    
    # Memory-intensive operations
    training_semaphore = asyncio.Semaphore(2)          # Background training
```

### Dynamic Resource Allocation

```python
class ResourceAllocator:
    async def allocate(self, task: AsyncTask):
        """Dynamically allocate resources based on task type and system load"""
        
        if task.priority == TaskPriority.USER_WAITING:
            # Immediate allocation for user-waiting tasks
            return await self.immediate_allocation(task)
            
        elif self.system_load() > 0.8:
            # High load: defer low-priority tasks
            if task.priority >= TaskPriority.VERIFICATION:
                await self.defer_task(task, delay=5.0)
                
        # Standard allocation based on task type
        if task.task_type == TaskType.COLUMN_INFERENCE:
            async with self.gpu_inference_semaphore:
                return await self.execute_task(task)
```

## Parallel Column Query System

### Column Query Orchestration

```python
class ColumnQueryOrchestrator:
    async def parallel_query(self, input_text: str, priority: TaskPriority):
        """Execute parallel queries across all columns"""
        
        tasks = []
        for column_id in range(4):
            task = AsyncTask(
                task_id=f"col_query_{uuid4()}",
                priority=priority,
                task_type=TaskType.COLUMN_INFERENCE,
                payload={
                    "column_id": column_id,
                    "input": input_text,
                    "temperature": 0.7 if column_id > 0 else 0.0  # Base column deterministic
                }
            )
            tasks.append(self.queue_task(task))
        
        # Wait for all columns with appropriate timeout
        timeout = 2.0 if priority == TaskPriority.USER_WAITING else 10.0
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            return self.synthesize_column_responses(results)
            
        except asyncio.TimeoutError:
            # Return partial results
            completed = [t for t in tasks if t.done()]
            return self.synthesize_partial_responses(completed)
```

### Response Synthesis

```python
class ResponseSynthesizer:
    def synthesize_column_responses(self, responses: List[ColumnResponse]):
        """Combine multiple column outputs into final response"""
        
        # Calculate agreement metrics
        variance = self.calculate_variance(responses)
        entropy = self.calculate_ensemble_entropy(responses)
        
        # High agreement: weighted average
        if variance < self.variance_threshold:
            weights = self.calculate_confidence_weights(responses)
            return self.weighted_average(responses, weights)
        
        # Disagreement: trigger uncertainty handling
        else:
            return UncertaintyResponse(
                base_response=responses[0],  # Base column as fallback
                uncertainty_score=variance,
                column_disagreements=self.extract_disagreements(responses),
                requires_clarification=variance > self.clarification_threshold
            )
```

## Tool Integration System

### External Tool Manager

```python
class ToolManager:
    registered_tools = {
        "wolfram": WolframAlphaAPI(),
        "search": SearchEngineAPI(),
        "calculator": LocalCalculator(),
        "code_exec": SandboxedExecutor(),
        "fact_check": FactCheckAPI()
    }
    
    async def research_with_tools(self, query: str, priority: TaskPriority):
        """Execute tool-based research for uncertain queries"""
        
        # Determine relevant tools
        relevant_tools = self.select_tools(query)
        
        # Create tasks for each tool
        tool_tasks = []
        for tool_name in relevant_tools:
            task = AsyncTask(
                task_id=f"tool_{tool_name}_{uuid4()}",
                priority=priority,
                task_type=TaskType.TOOL_CALL,
                payload={
                    "tool": tool_name,
                    "query": self.format_query_for_tool(query, tool_name)
                },
                timeout=self.get_tool_timeout(tool_name)
            )
            tool_tasks.append(self.queue_task(task))
        
        # Gather results
        results = await asyncio.gather(*tool_tasks, return_exceptions=True)
        return self.aggregate_tool_results(results)
```

## Verification Pipeline

### Multi-Encoder Verification

```python
class VerificationPipeline:
    encoders = {
        "bias": BiasDetector(),
        "logic": LogicProofChecker(),
        "sophism": SophismAnalyzer(),
        "test": TestExecutor(),
        "implication": ImplicationCoefficientCalculator()
    }
    
    async def verify_response(self, response: str, context: Dict, priority: TaskPriority):
        """Run response through verification encoders"""
        
        verification_tasks = []
        for encoder_name, encoder in self.encoders.items():
            task = AsyncTask(
                task_id=f"verify_{encoder_name}_{uuid4()}",
                priority=priority,
                task_type=TaskType.VERIFICATION,
                payload={
                    "encoder": encoder_name,
                    "response": response,
                    "context": context
                },
                timeout=2.0  # Quick verification
            )
            verification_tasks.append(self.queue_task(task))
        
        results = await asyncio.gather(*verification_tasks)
        
        return VerificationResult(
            bias_score=results[0],
            logic_valid=results[1],
            sophism_detected=results[2],
            tests_passed=results[3],
            implication_coefficient=results[4],
            overall_confidence=self.calculate_overall_confidence(results)
        )
```

## Priority Escalation System

### Dynamic Priority Adjustment

```python
class PriorityEscalator:
    async def monitor_task_age(self):
        """Background process that escalates old tasks"""
        while True:
            current_time = time.time()
            
            for task in self.pending_tasks:
                age = current_time - task.created_at
                
                # Escalate if waiting too long
                if age > self.escalation_thresholds[task.priority]:
                    await self.escalate_priority(task)
                    
                # User started waiting for background task
                if task.user_session_id and self.user_is_waiting(task.user_session_id):
                    task.priority = TaskPriority.USER_WAITING
                    await self.requeue_task(task)
                    
            await asyncio.sleep(0.5)  # Check every 500ms
    
    def escalation_thresholds(self):
        return {
            TaskPriority.CLARIFICATION: 5.0,      # 5 seconds
            TaskPriority.TOOL_RESEARCH: 10.0,     # 10 seconds
            TaskPriority.COLUMN_QUERY: 15.0,      # 15 seconds
            TaskPriority.VERIFICATION: 20.0,      # 20 seconds
            TaskPriority.BACKGROUND_TRAIN: 60.0   # 1 minute
        }
```

## Graceful Degradation

### Timeout Handling

```python
class TimeoutManager:
    async def execute_with_fallback(self, task: AsyncTask):
        """Execute task with timeout and fallback strategies"""
        
        try:
            # Try primary execution
            result = await asyncio.wait_for(
                self.execute_task(task),
                timeout=task.timeout
            )
            return result
            
        except asyncio.TimeoutError:
            # Try fallback based on task type
            if task.task_type == TaskType.COLUMN_INFERENCE:
                # Return base column result only
                return await self.get_base_column_cached(task.payload["input"])
                
            elif task.task_type == TaskType.TOOL_CALL:
                # Return without external verification
                return self.generate_uncertain_response(task.payload["query"])
                
            elif task.task_type == TaskType.VERIFICATION:
                # Skip verification, mark as unverified
                return VerificationResult(verified=False)
```

### Partial Result Assembly

```python
class PartialResultHandler:
    def assemble_partial_response(self, completed_tasks: List[AsyncTask]):
        """Build best possible response from partial results"""
        
        # Group by task type
        column_results = [t.result for t in completed_tasks 
                         if t.task_type == TaskType.COLUMN_INFERENCE and t.result]
        tool_results = [t.result for t in completed_tasks 
                       if t.task_type == TaskType.TOOL_CALL and t.result]
        
        # At minimum, need base column
        if not any(t.payload.get("column_id") == 0 for t in completed_tasks):
            return ErrorResponse("Insufficient data for response")
        
        # Build response with confidence indicators
        response = PartialResponse(
            text=self.synthesize_available(column_results, tool_results),
            completeness=len(completed_tasks) / self.expected_task_count,
            missing_components=self.identify_missing(completed_tasks),
            confidence=self.calculate_partial_confidence(completed_tasks)
        )
        
        return response
```

## Monitoring & Metrics

### Performance Metrics

```python
@dataclass
class OrchestrationMetrics:
    # Latency metrics
    avg_user_latency: float          # Average time for USER_WAITING tasks
    p95_user_latency: float          # 95th percentile user latency
    avg_background_latency: float    # Average time for background tasks
    
    # Throughput metrics
    tasks_per_second: float          # Overall throughput
    gpu_utilization: float           # GPU usage percentage
    api_calls_per_minute: Dict[str, float]  # Per-API usage
    
    # Quality metrics
    timeout_rate: float              # Percentage of timeouts
    error_rate: float                # Percentage of errors
    partial_response_rate: float    # Percentage of partial responses
    
    # Queue metrics
    queue_depth: Dict[TaskPriority, int]  # Tasks per priority level
    avg_queue_time: Dict[TaskPriority, float]  # Wait time per priority
    
    # Resource metrics
    semaphore_contention: Dict[str, float]  # Contention per resource
    worker_utilization: float        # Worker thread usage
```

### Health Monitoring

```python
class HealthMonitor:
    async def monitor_system_health(self):
        """Continuous health monitoring"""
        
        while True:
            metrics = await self.collect_metrics()
            
            # Alert on degradation
            if metrics.avg_user_latency > self.latency_threshold:
                await self.alert("High user latency", metrics)
                
            if metrics.timeout_rate > 0.1:  # >10% timeouts
                await self.alert("High timeout rate", metrics)
                
            if metrics.queue_depth[TaskPriority.USER_WAITING] > 10:
                await self.scale_workers()  # Dynamic scaling
                
            # Log metrics
            await self.log_metrics(metrics)
            
            await asyncio.sleep(1.0)  # Check every second
```

## Configuration

### System Configuration

```yaml
orchestration:
  # Worker configuration
  max_workers: 16
  min_workers: 4
  worker_scaling_enabled: true
  
  # Resource limits
  max_gpu_parallel: 4
  max_api_concurrent: 20
  max_memory_gb: 32
  
  # Timeouts (seconds)
  user_waiting_timeout: 5.0
  background_timeout: 60.0
  tool_timeout: 10.0
  
  # Priority escalation
  escalation_enabled: true
  escalation_check_interval: 0.5
  
  # Graceful degradation
  partial_results_enabled: true
  fallback_to_cache: true
  
  # Monitoring
  metrics_interval: 1.0
  alert_webhook: "https://monitoring.example.com/alerts"
```

## Error Handling

### Retry Logic

```python
class RetryHandler:
    async def execute_with_retry(self, task: AsyncTask):
        """Execute task with exponential backoff retry"""
        
        for attempt in range(task.retries):
            try:
                result = await self.execute_task(task)
                return result
                
            except Exception as e:
                if attempt == task.retries - 1:
                    # Final attempt failed
                    task.error = e
                    return ErrorResult(str(e))
                    
                # Exponential backoff
                delay = 2 ** attempt * 0.5
                await asyncio.sleep(delay)
                
                # Reduce priority on retry (avoid blocking high-priority queue)
                if task.priority < TaskPriority.BACKGROUND_TRAIN:
                    task.priority = TaskPriority(task.priority + 1)
```

## Integration Points

### Main Inference Loop Integration

```python
class InferenceOrchestrator:
    async def process_user_query(self, query: str, session_id: str):
        """Main entry point for user queries"""
        
        # Create context
        context = InferenceContext(
            query=query,
            session_id=session_id,
            timestamp=time.time()
        )
        
        # Phase 1: Parallel column query
        column_task = self.create_column_query_task(query, TaskPriority.USER_WAITING)
        
        # Phase 2: Uncertainty detection (from initial forward pass)
        initial_response, uncertainty = await self.quick_inference(query)
        
        if uncertainty > self.uncertainty_threshold:
            # Phase 3: Tool research (if uncertain)
            tool_task = self.create_tool_research_task(query, TaskPriority.USER_WAITING)
            
            # Phase 4: Wait for all results
            column_results, tool_results = await asyncio.gather(
                column_task,
                tool_task
            )
            
            # Phase 5: Synthesis
            response = self.synthesize(column_results, tool_results)
            
            # Phase 6: Verification (async, don't wait)
            asyncio.create_task(
                self.verify_response_async(response, context)
            )
            
            # Phase 7: Queue for training (background)
            await self.queue_training_sample(query, response, uncertainty)
            
        else:
            response = initial_response
        
        return response
```

## Future Enhancements

### Planned Features

1. **Predictive Prefetching**: Anticipate likely follow-up queries
2. **Result Caching**: Cache frequent queries with TTL
3. **Distributed Orchestration**: Multi-node task distribution
4. **Adaptive Timeouts**: Learn optimal timeouts per task type
5. **Smart Batching**: Batch similar tasks for efficiency
6. **Circuit Breakers**: Prevent cascade failures
7. **Task Dependencies**: DAG-based task dependencies
8. **Priority Inheritance**: Child tasks inherit parent priority

## Notes

This orchestration system is designed to:
- Prioritize user experience above all else
- Maximize resource utilization during idle periods
- Degrade gracefully under load
- Provide comprehensive monitoring and alerting
- Scale horizontally when needed

The async nature ensures the system remains responsive even under heavy load, while the priority system ensures critical user-facing operations always get resources first.
