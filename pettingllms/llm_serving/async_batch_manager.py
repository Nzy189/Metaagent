"""
Async Batch Manager for managing producer-consumer pipeline in rollout serving.
"""
import asyncio
import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np
from verl import DataProto
from tensordict import TensorDict


class BatchStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchRequest:
    """Represents a batch request for rollout generation"""
    batch_id: str
    env_outputs: List[Dict]
    mode: str  # "train" or "validation"
    status: BatchStatus = BatchStatus.PENDING
    created_at: float = None
    completed_at: float = None
    result: Optional[DataProto] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class AsyncBatchManager:
    """
    Manages asynchronous batch processing for rollout generation.
    Coordinates between producers and consumers using queues.
    """
    
    def __init__(self, 
                 max_queue_size: int = 100,
                 batch_timeout: float = 30.0,
                 max_concurrent_batches: int = 10):
        """
        Initialize the async batch manager.
        
        Args:
            max_queue_size: Maximum size of the request queue
            batch_timeout: Timeout for batch processing in seconds
            max_concurrent_batches: Maximum number of concurrent batches
        """
        self.max_queue_size = max_queue_size
        self.batch_timeout = batch_timeout
        self.max_concurrent_batches = max_concurrent_batches
        
        # Queues for producer-consumer pipeline
        self.pending_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_queue = asyncio.Queue(maxsize=max_concurrent_batches)
        self.completed_queue = asyncio.Queue()
        
        # Batch tracking
        self.active_batches: Dict[str, BatchRequest] = {}
        self.batch_lock = threading.Lock()
        
        # Control flags
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
    async def submit_batch(self, env_outputs: List[Dict], mode: str = "validation") -> str:
        """
        Submit a new batch request for processing.
        
        Args:
            env_outputs: Environment outputs to process
            mode: Processing mode ("train" or "validation")
            
        Returns:
            batch_id: Unique identifier for the batch
        """
        batch_id = str(uuid.uuid4())
        batch_request = BatchRequest(
            batch_id=batch_id,
            env_outputs=env_outputs,
            mode=mode
        )
        
        with self.batch_lock:
            self.active_batches[batch_id] = batch_request
        
        try:
            await asyncio.wait_for(
                self.pending_queue.put(batch_request),
                timeout=1.0
            )
            return batch_id
        except asyncio.TimeoutError:
            with self.batch_lock:
                del self.active_batches[batch_id]
            raise RuntimeError("Queue is full, cannot submit new batch")
    
    async def get_batch_result(self, batch_id: str, timeout: Optional[float] = None) -> DataProto:
        """
        Get the result of a batch request.
        
        Args:
            batch_id: Batch identifier
            timeout: Timeout for waiting for result
            
        Returns:
            result: DataProto containing the rollout batch
        """
        if timeout is None:
            timeout = self.batch_timeout
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.batch_lock:
                if batch_id not in self.active_batches:
                    raise ValueError(f"Batch {batch_id} not found")
                
                batch = self.active_batches[batch_id]
                if batch.status == BatchStatus.COMPLETED:
                    result = batch.result
                    del self.active_batches[batch_id]
                    return result
                elif batch.status == BatchStatus.FAILED:
                    error = batch.error
                    del self.active_batches[batch_id]
                    raise RuntimeError(f"Batch processing failed: {error}")
            
            await asyncio.sleep(0.1)
        
        # Timeout reached
        with self.batch_lock:
            if batch_id in self.active_batches:
                del self.active_batches[batch_id]
        raise TimeoutError(f"Batch {batch_id} processing timed out")
    
    async def get_pending_batch(self) -> Optional[BatchRequest]:
        """Get the next pending batch for processing."""
        if self.shutdown_event.is_set():
            return None
            
        try:
            batch = await asyncio.wait_for(
                self.pending_queue.get(),
                timeout=1.0
            )
            batch.status = BatchStatus.PROCESSING
            await self.processing_queue.put(batch)
            return batch
        except asyncio.TimeoutError:
            return None
    
    async def get_processing_batch(self) -> Optional[BatchRequest]:
        """Get the next batch that's ready for processing."""
        if self.shutdown_event.is_set():
            return None
            
        try:
            return await asyncio.wait_for(
                self.processing_queue.get(),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            return None
    
    async def complete_batch(self, batch_id: str, result: DataProto):
        """Mark a batch as completed with result."""
        with self.batch_lock:
            if batch_id in self.active_batches:
                batch = self.active_batches[batch_id]
                batch.status = BatchStatus.COMPLETED
                batch.result = result
                batch.completed_at = time.time()
    
    async def fail_batch(self, batch_id: str, error: str):
        """Mark a batch as failed with error message."""
        with self.batch_lock:
            if batch_id in self.active_batches:
                batch = self.active_batches[batch_id]
                batch.status = BatchStatus.FAILED
                batch.error = error
                batch.completed_at = time.time()
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchStatus]:
        """Get the current status of a batch."""
        with self.batch_lock:
            if batch_id in self.active_batches:
                return self.active_batches[batch_id].status
        return None
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics about queue sizes and active batches."""
        with self.batch_lock:
            return {
                "pending_queue_size": self.pending_queue.qsize(),
                "processing_queue_size": self.processing_queue.qsize(),
                "completed_queue_size": self.completed_queue.qsize(),
                "active_batches": len(self.active_batches),
                "pending_batches": sum(1 for b in self.active_batches.values() if b.status == BatchStatus.PENDING),
                "processing_batches": sum(1 for b in self.active_batches.values() if b.status == BatchStatus.PROCESSING),
                "completed_batches": sum(1 for b in self.active_batches.values() if b.status == BatchStatus.COMPLETED),
                "failed_batches": sum(1 for b in self.active_batches.values() if b.status == BatchStatus.FAILED),
            }
    
    async def start(self):
        """Start the batch manager."""
        self.is_running = True
        self.shutdown_event.clear()
    
    async def shutdown(self):
        """Shutdown the batch manager."""
        self.is_running = False
        self.shutdown_event.set()
        
        # Wait for all active batches to complete or timeout
        timeout = 5.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.batch_lock:
                active_count = sum(
                    1 for b in self.active_batches.values() 
                    if b.status in [BatchStatus.PENDING, BatchStatus.PROCESSING]
                )
                if active_count == 0:
                    break
            await asyncio.sleep(0.1)
        
        # Clean up any remaining batches
        with self.batch_lock:
            for batch in self.active_batches.values():
                if batch.status in [BatchStatus.PENDING, BatchStatus.PROCESSING]:
                    batch.status = BatchStatus.FAILED
                    batch.error = "Shutdown" 