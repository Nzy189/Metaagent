"""
Request Buffer for Producer-Consumer mechanism in multi-turn rollouts.
This module handles the buffering and coordination between environment state production
and LLM response consumption for efficient rollout processing.

Author: AI Assistant
Date: 2025-01-XX
"""

import queue
import threading
import time
from typing import List, Dict, Any, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from verl import DataProto
else:
    try:
        from verl import DataProto
    except ImportError:
        DataProto = Any


class RequestStatus(Enum):
    """Status of a request in the buffer"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TurnRequest:
    """A single turn request containing environment state and context"""
    turn_id: int
    env_outputs: List[Dict]
    timestamp: float
    status: RequestStatus = RequestStatus.PENDING
    lm_inputs: Optional[DataProto] = None
    lm_outputs: Optional[DataProto] = None
    env_inputs: Optional[List[Dict]] = None
    error_message: Optional[str] = None


@dataclass
class RolloutSession:
    """A complete rollout session containing multiple turns"""
    session_id: str
    mode: str  # "train" or "validation"
    max_turns: int
    current_turn: int = 0
    requests: List[TurnRequest] = field(default_factory=list)
    is_completed: bool = False
    final_rollouts: Optional[DataProto] = None


class RequestBuffer:
    """
    Producer-Consumer buffer for managing multi-turn rollout requests.
    
    The buffer coordinates between:
    - Producer: Generates prompts from environment states
    - Consumer: Processes LLM responses and updates environments
    """
    
    def __init__(self, max_buffer_size: int = 100, max_concurrent_requests: int = 10):
        """
        Initialize the request buffer.
        
        Args:
            max_buffer_size: Maximum number of requests in buffer
            max_concurrent_requests: Maximum concurrent processing requests
        """
        self.max_buffer_size = max_buffer_size
        self.max_concurrent_requests = max_concurrent_requests
        
        # Thread-safe queues for different stages
        self.prompt_queue = queue.Queue(maxsize=max_buffer_size)  # env_outputs -> lm_inputs
        self.generation_queue = queue.Queue(maxsize=max_buffer_size)  # lm_inputs -> lm_outputs
        self.action_queue = queue.Queue(maxsize=max_buffer_size)  # lm_outputs -> env_inputs
        
        # Session management
        self.active_sessions: Dict[str, RolloutSession] = {}
        self.completed_sessions: Dict[str, RolloutSession] = {}
        
        # Thread synchronization
        self._lock = threading.Lock()
        self._shutdown = False
        self._active_threads = []
        
    def create_session(self, session_id: str, mode: str, max_turns: int) -> RolloutSession:
        """Create a new rollout session"""
        with self._lock:
            if session_id in self.active_sessions:
                raise ValueError(f"Session {session_id} already exists")
            
            session = RolloutSession(
                session_id=session_id,
                mode=mode,
                max_turns=max_turns
            )
            self.active_sessions[session_id] = session
            return session
    
    def add_turn_request(self, session_id: str, env_outputs: List[Dict]) -> TurnRequest:
        """
        Add a new turn request to the buffer (Producer step).
        
        Args:
            session_id: Unique session identifier
            env_outputs: Environment outputs for current turn
            
        Returns:
            TurnRequest object
        """
        with self._lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            if session.is_completed:
                raise ValueError(f"Session {session_id} is already completed")
            
            turn_request = TurnRequest(
                turn_id=session.current_turn,
                env_outputs=env_outputs,
                timestamp=time.time()
            )
            
            session.requests.append(turn_request)
            session.current_turn += 1
            
            # Add to prompt queue for processing
            try:
                self.prompt_queue.put_nowait((session_id, turn_request))
            except queue.Full:
                turn_request.status = RequestStatus.FAILED
                turn_request.error_message = "Buffer queue is full"
                raise RuntimeError("Request buffer is full")
            
            return turn_request
    
    def get_pending_prompts(self, timeout: Optional[float] = None) -> List[tuple]:
        """
        Get pending prompt requests for batch processing.
        
        Args:
            timeout: Maximum time to wait for requests
            
        Returns:
            List of (session_id, turn_request) tuples
        """
        requests = []
        end_time = time.time() + (timeout or 0)
        
        while len(requests) < self.max_concurrent_requests:
            try:
                remaining_time = max(0, end_time - time.time()) if timeout else 1.0
                if remaining_time <= 0 and timeout:
                    break
                    
                session_id, turn_request = self.prompt_queue.get(
                    timeout=remaining_time if timeout else False
                )
                turn_request.status = RequestStatus.PROCESSING
                requests.append((session_id, turn_request))
                
            except queue.Empty:
                break
                
        return requests
    
    def submit_lm_inputs(self, session_id: str, turn_id: int, lm_inputs: DataProto):
        """Submit processed LM inputs for generation"""
        with self._lock:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            if turn_id >= len(session.requests):
                raise ValueError(f"Turn {turn_id} not found in session {session_id}")
            
            turn_request = session.requests[turn_id]
            turn_request.lm_inputs = lm_inputs
            
            try:
                self.generation_queue.put_nowait((session_id, turn_request))
            except queue.Full:
                turn_request.status = RequestStatus.FAILED
                turn_request.error_message = "Generation queue is full"
                raise RuntimeError("Generation queue is full")
    
    def get_pending_generations(self, timeout: Optional[float] = None) -> List[tuple]:
        """Get pending generation requests for batch processing"""
        requests = []
        end_time = time.time() + (timeout or 0)
        
        while len(requests) < self.max_concurrent_requests:
            try:
                remaining_time = max(0, end_time - time.time()) if timeout else 1.0
                if remaining_time <= 0 and timeout:
                    break
                    
                session_id, turn_request = self.generation_queue.get(
                    timeout=remaining_time if timeout else False
                )
                requests.append((session_id, turn_request))
                
            except queue.Empty:
                break
                
        return requests
    
    def submit_lm_outputs(self, session_id: str, turn_id: int, lm_outputs: DataProto):
        """Submit LM generation outputs for environment processing"""
        with self._lock:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            if turn_id >= len(session.requests):
                raise ValueError(f"Turn {turn_id} not found in session {session_id}")
            
            turn_request = session.requests[turn_id]
            turn_request.lm_outputs = lm_outputs
            
            try:
                self.action_queue.put_nowait((session_id, turn_request))
            except queue.Full:
                turn_request.status = RequestStatus.FAILED
                turn_request.error_message = "Action queue is full"
                raise RuntimeError("Action queue is full")
    
    def get_pending_actions(self, timeout: Optional[float] = None) -> List[tuple]:
        """Get pending action requests for environment processing"""
        requests = []
        end_time = time.time() + (timeout or 0)
        
        while len(requests) < self.max_concurrent_requests:
            try:
                remaining_time = max(0, end_time - time.time()) if timeout else 1.0
                if remaining_time <= 0 and timeout:
                    break
                    
                session_id, turn_request = self.action_queue.get(
                    timeout=remaining_time if timeout else False
                )
                requests.append((session_id, turn_request))
                
            except queue.Empty:
                break
                
        return requests
    
    def complete_turn(self, session_id: str, turn_id: int, env_inputs: List[Dict]):
        """Mark a turn as completed and update session state"""
        with self._lock:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            if turn_id >= len(session.requests):
                raise ValueError(f"Turn {turn_id} not found in session {session_id}")
            
            turn_request = session.requests[turn_id]
            turn_request.env_inputs = env_inputs
            turn_request.status = RequestStatus.COMPLETED
    
    def complete_session(self, session_id: str, final_rollouts: DataProto):
        """Mark a session as completed and move to completed sessions"""
        with self._lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            session.is_completed = True
            session.final_rollouts = final_rollouts
            
            # Move to completed sessions
            self.completed_sessions[session_id] = session
            del self.active_sessions[session_id]
    
    def get_session_status(self, session_id: str) -> Optional[RolloutSession]:
        """Get current status of a session"""
        with self._lock:
            return (self.active_sessions.get(session_id) or 
                    self.completed_sessions.get(session_id))
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes for monitoring"""
        return {
            "prompt_queue": self.prompt_queue.qsize(),
            "generation_queue": self.generation_queue.qsize(),
            "action_queue": self.action_queue.qsize(),
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.completed_sessions)
        }
    
    def clear_completed_sessions(self, older_than_seconds: Optional[float] = None):
        """Clear completed sessions to free memory"""
        with self._lock:
            if older_than_seconds is None:
                self.completed_sessions.clear()
            else:
                current_time = time.time()
                to_remove = [
                    session_id for session_id, session in self.completed_sessions.items()
                    if any(req.timestamp < current_time - older_than_seconds 
                          for req in session.requests)
                ]
                for session_id in to_remove:
                    del self.completed_sessions[session_id]
    
    def shutdown(self):
        """Shutdown the buffer and clean up resources"""
        self._shutdown = True
        
        # Clear all queues
        while not self.prompt_queue.empty():
            try:
                self.prompt_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.generation_queue.empty():
            try:
                self.generation_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear sessions
        with self._lock:
            self.active_sessions.clear()
            self.completed_sessions.clear()


class AsyncRolloutProcessor:
    """
    Asynchronous processor for handling rollout requests using the RequestBuffer.
    Coordinates the producer-consumer flow for multi-turn rollouts.
    """
    
    def __init__(self, 
                 buffer: RequestBuffer,
                 ctx_manager_producer: Callable,
                 llm_generator: Callable,
                 ctx_manager_consumer: Callable,
                 es_manager: Callable):
        """
        Initialize the async processor.
        
        Args:
            buffer: RequestBuffer instance
            ctx_manager_producer: Function to generate LM inputs from env outputs
            llm_generator: Function to generate LM outputs from LM inputs  
            ctx_manager_consumer: Function to generate env inputs from LM outputs
            es_manager: Function to process env inputs and get new env outputs
        """
        self.buffer = buffer
        self.ctx_manager_producer = ctx_manager_producer
        self.llm_generator = llm_generator
        self.ctx_manager_consumer = ctx_manager_consumer
        self.es_manager = es_manager
        
        self._worker_threads = []
        self._shutdown = False
    
    def start_workers(self, num_workers: int = 3):
        """Start worker threads for processing different stages"""
        self._worker_threads = [
            threading.Thread(target=self._prompt_worker, daemon=True),
            threading.Thread(target=self._generation_worker, daemon=True),
            threading.Thread(target=self._action_worker, daemon=True)
        ]
        
        for thread in self._worker_threads:
            thread.start()
    
    def _prompt_worker(self):
        """Worker thread for processing prompt generation"""
        while not self._shutdown:
            try:
                requests = self.buffer.get_pending_prompts(timeout=1.0)
                if not requests:
                    continue
                
                # Batch process prompts
                for session_id, turn_request in requests:
                    try:
                        lm_inputs = self.ctx_manager_producer(
                            turn_request.env_outputs, 
                            prepare_for_update=False
                        )
                        self.buffer.submit_lm_inputs(
                            session_id, 
                            turn_request.turn_id, 
                            lm_inputs
                        )
                    except Exception as e:
                        turn_request.status = RequestStatus.FAILED
                        turn_request.error_message = str(e)
                        
            except Exception as e:
                print(f"Error in prompt worker: {e}")
    
    def _generation_worker(self):
        """Worker thread for processing LLM generation"""
        while not self._shutdown:
            try:
                requests = self.buffer.get_pending_generations(timeout=1.0)
                if not requests:
                    continue
                
                # Batch process generations
                batch_inputs = []
                request_info = []
                
                for session_id, turn_request in requests:
                    if turn_request.lm_inputs:
                        batch_inputs.append(turn_request.lm_inputs)
                        request_info.append((session_id, turn_request))
                
                if batch_inputs:
                    try:
                        # Process batch generation
                        batch_outputs = self.llm_generator(batch_inputs)
                        
                        # Submit results
                        for (session_id, turn_request), lm_outputs in zip(request_info, batch_outputs):
                            self.buffer.submit_lm_outputs(
                                session_id,
                                turn_request.turn_id,
                                lm_outputs
                            )
                    except Exception as e:
                        for session_id, turn_request in request_info:
                            turn_request.status = RequestStatus.FAILED
                            turn_request.error_message = str(e)
                        
            except Exception as e:
                print(f"Error in generation worker: {e}")
    
    def _action_worker(self):
        """Worker thread for processing environment actions"""
        while not self._shutdown:
            try:
                requests = self.buffer.get_pending_actions(timeout=1.0)
                if not requests:
                    continue
                
                # Process actions
                for session_id, turn_request in requests:
                    try:
                        if turn_request.lm_outputs:
                            env_inputs = self.ctx_manager_consumer(turn_request.lm_outputs)
                            self.buffer.complete_turn(
                                session_id,
                                turn_request.turn_id,
                                env_inputs
                            )
                    except Exception as e:
                        turn_request.status = RequestStatus.FAILED
                        turn_request.error_message = str(e)
                        
            except Exception as e:
                print(f"Error in action worker: {e}")
    
    def shutdown(self):
        """Shutdown the processor and wait for workers to finish"""
        self._shutdown = True
        for thread in self._worker_threads:
            thread.join(timeout=5.0)
        self.buffer.shutdown()
