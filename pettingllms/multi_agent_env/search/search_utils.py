import logging
import re
import json
import os
import time
import uuid
import threading
from typing import Any, Dict, List, Optional, Union
import random
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Constants for API calls
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1

def truncatefn(s, length=300):
    """Truncate text to specified length while preserving context."""
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= length:
        return s
    return s[:length//2] + "...(truncated)..." + s[-length//2:]

def extract_search_query_from_response(response: str) -> str:
    """Extract search query from agent response."""
    # Look for patterns like "**Search Query:**" or "Search:"
    patterns = [
        r"\*\*Search Query:\*\*\s*(.+)",
        r"Search Query:\s*(.+)",
        r"search:\s*(.+)",
        r"query:\s*(.+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    # Fallback: take first line if no pattern matches
    lines = response.strip().split('\n')
    if lines:
        return lines[0].strip()
    
    return response.strip()

def extract_answer_from_response(response: str) -> Optional[str]:
    """Extract final answer from agent response."""
    # Look for answer patterns
    patterns = [
        r"\*\*Final Answer:\*\*\s*(.+)",
        r"\*\*Answer:\*\*\s*(.+)",
        r"Final Answer:\s*(.+)",
        r"Answer:\s*(.+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    return None

def format_search_results(search_results: List[Dict[str, Any]], max_results: int = 5) -> str:
    """Format search results for display to agents."""
    if not search_results:
        return "No search results found."
    
    formatted = "Search Results:\n"
    for i, result in enumerate(search_results[:max_results], 1):
        title = result.get('title', 'No title')
        snippet = result.get('snippet', result.get('content', 'No content'))
        url = result.get('url', '')
        
        formatted += f"{i}. {title}\n"
        formatted += f"   {truncatefn(snippet, 200)}\n"
        if url:
            formatted += f"   URL: {url}\n"
        formatted += "\n"
    
    return formatted

def evaluate_search_answer(predicted_answer: str, ground_truth: str) -> bool:
    """Evaluate if the predicted answer matches the ground truth."""
    if not predicted_answer or not ground_truth:
        return False
    
    # Normalize for comparison
    pred_norm = predicted_answer.lower().strip()
    truth_norm = ground_truth.lower().strip()
    
    # Exact match
    if pred_norm == truth_norm:
        return True
    
    # Contains match
    if pred_norm in truth_norm or truth_norm in pred_norm:
        return True
    
    # Word overlap (for more complex answers)
    pred_words = set(pred_norm.split())
    truth_words = set(truth_norm.split())
    
    if len(pred_words & truth_words) >= min(len(pred_words), len(truth_words)) * 0.5:
        return True
    
    return False

def load_search_problem_batch(
    env_indices: List[int],
    dataset_name: str = "bamboogle",
    mode: str = "train",
    config: dict = None
) -> List[Dict[str, Any]]:
    """
    Load a batch of search problems.
    
    Args:
        env_indices: List of environment indices
        dataset_name: Name of the benchmark for test mode (e.g., "bamboogle", "2wiki", "hotpotqa", "musique")
                     Ignored for train mode (always uses train.parquet)
        mode: "train" or "validate"
        config: Configuration dict (unused, kept for compatibility)
        
    Returns:
        A list of dicts with keys: question, ground_truth, search_hint
    """
    try:
        import datasets
        from pathlib import Path
        DATASETS_AVAILABLE = True
    except ImportError:
        raise ImportError("datasets library is required for loading search problems")
    
    # Determine dataset directory and file path
    current_dir = Path(__file__).parent.parent.parent.parent
    local_datasets_dir = current_dir / "data" / "search"
    
    if mode == "train":
        # Training mode: load from train/train.parquet
        parquet_file = local_datasets_dir / "train" / "train.parquet"
    else:
        # Validation/test mode: load from test/{benchmark_name}.parquet
        parquet_file = local_datasets_dir / "test" / f"{dataset_name}.parquet"
    
    # Check if dataset file exists
    if not parquet_file.exists():
        logger.warning(f"Dataset file not found: {parquet_file}")
        # Return mock data for testing if dataset not found
        return _generate_mock_search_problems(env_indices, mode)
    
    logger.info(f"Loading search dataset from: {parquet_file}")
    
    try:
        ds = datasets.load_dataset("parquet", data_files=str(parquet_file), split="train")
        logger.info(f"Search dataset loaded: {len(ds)} samples")
    except Exception as e:
        logger.warning(f"Failed to load dataset from {parquet_file}: {e}")
        return _generate_mock_search_problems(env_indices, mode)
    
    batch_results = []
    
    if mode == "train":
        # Training mode: sample from dataset based on env_indices
        if len(ds) < len(env_indices):
            logger.warning(f"Dataset has {len(ds)} samples, but {len(env_indices)} requested")
            # Repeat dataset if needed
            indices = [i % len(ds) for i in env_indices]
        else:
            indices = [i % len(ds) for i in env_indices]
        
        for idx in indices:
            problem_dict = _format_search_problem(ds[idx], idx, mode="train")
            if problem_dict:
                batch_results.append(problem_dict)
    else:
        # Validation mode: use all samples or subset based on env_indices
        max_samples = min(len(ds), len(env_indices) if env_indices else len(ds))
        for i in range(max_samples):
            example = ds[i]
            problem_dict = _format_search_problem(example, i, mode="validate")
            if problem_dict:
                batch_results.append(problem_dict)
    
    logger.info(f"Returning {len(batch_results)} search problems")
    return batch_results


def _format_search_problem(example: Dict, index: int, mode: str = "train") -> Optional[Dict]:
    """
    Format a search problem example into a standardized dictionary.
    
    Args:
        example: Raw example from dataset
        index: Index of the example
        mode: "train" or "validate"
        
    Returns:
        Formatted problem dictionary or None if invalid
    """
    try:
        question = None
        ground_truth = None
        search_hint = ""
        
        # Try different data formats
        if mode == "train":
            # Training format: {id, question, chain, result, source, extra_info}
            question = example.get("question", "")
            ground_truth = example.get("result", "")
            
            # Try to get from extra_info if not found
            if not ground_truth and "extra_info" in example:
                extra_info = example["extra_info"]
                if isinstance(extra_info, dict):
                    ground_truth = extra_info.get("ground_truth", "")
        else:
            # Test format: {data_source, prompt, ability, reward_model, extra_info}
            # Extract question from prompt content
            if "prompt" in example and isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
                question = example["prompt"][0].get("content", "")
            elif "question" in example:
                question = example.get("question", "")
            
            # Extract ground truth from reward_model
            if "reward_model" in example and "ground_truth" in example["reward_model"]:
                gt_data = example["reward_model"]["ground_truth"]
                if isinstance(gt_data, dict) and "target" in gt_data:
                    ground_truth = gt_data.get("target", [])
                    if isinstance(ground_truth, list) and ground_truth:
                        # Join list items with "; " as specified in load_search.py
                        ground_truth = "; ".join([str(x) for x in ground_truth])
                elif isinstance(gt_data, (str, int, float)):
                    ground_truth = str(gt_data)
            elif "result" in example:
                ground_truth = example.get("result", "")
        
        # Handle golden_answers as fallback
        if not ground_truth and "golden_answers" in example:
            golden_ans = example.get("golden_answers", [])
            if isinstance(golden_ans, list) and golden_ans:
                ground_truth = "; ".join([str(x) for x in golden_ans])
            elif isinstance(golden_ans, str):
                ground_truth = golden_ans
        
        # Extract search hint if available
        search_hint = example.get("search_hint", "")
        
        if not question:
            logger.warning(f"Skipping example {index}: missing question field")
            return None
        
        if not ground_truth:
            logger.warning(f"Skipping example {index}: missing ground truth")
            return None
        
        return {
            "question": question.strip(),
            "ground_truth": str(ground_truth).strip(),
            "search_hint": search_hint.strip() if search_hint else ""
        }
        
    except Exception as e:
        logger.warning(f"Error formatting search example {index}: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return None


def _generate_mock_search_problems(env_indices: List[int], mode: str = "train") -> List[Dict[str, Any]]:
    """
    Generate mock search problems for testing when dataset is not available.
    
    Args:
        env_indices: List of environment indices
        mode: "train" or "validate"
        
    Returns:
        List of mock search problem dictionaries
    """
    mock_problems = [
        {
            "question": "What is the capital of France?",
            "ground_truth": "Paris",
            "search_hint": "Look for information about French geography and government."
        },
        {
            "question": "Who wrote the novel '1984'?",
            "ground_truth": "George Orwell",
            "search_hint": "Search for dystopian literature and British authors."
        },
        {
            "question": "What is the largest planet in our solar system?",
            "ground_truth": "Jupiter",
            "search_hint": "Search for planetary science and solar system facts."
        },
        {
            "question": "In what year did World War II end?",
            "ground_truth": "1945",
            "search_hint": "Look for historical events and timeline information."
        },
        {
            "question": "What is the chemical symbol for gold?",
            "ground_truth": "Au",
            "search_hint": "Search for chemistry and periodic table information."
        }
    ]
    
    batch_results = []
    num_problems = len(env_indices) if mode == "train" else min(len(mock_problems), len(env_indices) if env_indices else len(mock_problems))
    
    for i in range(num_problems):
        problem_idx = i % len(mock_problems)
        batch_results.append(mock_problems[problem_idx].copy())
    
    logger.info(f"Generated {len(batch_results)} mock search problems for testing")
    return batch_results

def clean_search_text(text: str) -> str:
    """Clean text for search processing."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common stop words for search optimization
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stop_words or len(words) <= 3]
    
    return ' '.join(cleaned_words)

def call_search_api(
    retrieval_service_url: str,
    query: str,
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    log_requests: bool = True,
    session: Optional[requests.Session] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Calls the search API with a single query.

    Args:
        retrieval_service_url: The URL of the search API.
        query: The query to search for.
        topk: The number of results to return.
        return_scores: Whether to return scores for the results.
        timeout: The timeout for the request.
        log_requests: Whether to log requests.
        session: The session to use for the request. If none is provided, a new session will be created.

    Returns:
        response: The response from the search API (json if successful, None otherwise)
        error_msg: The error message if the request failed.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {"query": query, "topk": topk, "return_scores": return_scores}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Use provided session or create a new one for this request
    if session is None:
        session = requests.Session()
        should_close_session = True
    else:
        should_close_session = False

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if log_requests:
                logger.info(
                    f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}"
                )
            response = session.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt + 1}/{MAX_RETRIES}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            if log_requests:
                logger.info(f"{log_prefix}Search API call successful on attempt {attempt + 1}")

            # Close session if we created it
            if should_close_session:
                session.close()

            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If all attempts failed
    logger.error(f"{log_prefix}API Request Failed after {MAX_RETRIES} attempts: {last_error}")

    # Close session if we created it
    if should_close_session:
        session.close()

    return None, last_error

def extract_sub_questions_from_response(response: str) -> List[str]:
    """Extract sub-questions from decomposition response."""
    # Look for patterns like "### question" or "Sub-question 1:"
    patterns = [
        r"###\s*(.+)",
        r"Sub-question\s+\d+:\s*(.+)",
        r"\d+\.\s*(.+)",
        r"-\s*(.+)"
    ]
    
    sub_questions = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                question = match.group(1).strip()
                if question and len(question) > 10:  # Filter out very short matches
                    sub_questions.append(question)
                break
    
    return sub_questions

def parse_sub_questions_with_references(decomposition_text: str) -> List[str]:
    """Parse sub-questions and handle references like #1, #2, etc."""
    sub_questions = extract_sub_questions_from_response(decomposition_text)
    
    # Process references in sub-questions
    processed_questions = []
    for i, question in enumerate(sub_questions):
        # Replace references like #1, #2 with actual previous answers
        processed_question = question
        for j in range(i):
            ref_pattern = f"#{j+1}"
            if ref_pattern in processed_question:
                processed_question = processed_question.replace(
                    ref_pattern, 
                    f"(answer from sub-question {j+1})"
                )
        processed_questions.append(processed_question)
    
    return processed_questions

def format_sub_answers_text(sub_questions: List[str], sub_answers: List[str]) -> str:
    """Format sub-questions and their answers for final reasoning."""
    if not sub_questions or not sub_answers:
        return ""
    
    formatted_text = "Sub-questions and their answers:\n"
    for i, (question, answer) in enumerate(zip(sub_questions, sub_answers), 1):
        formatted_text += f"{i}. {question}\n"
        formatted_text += f"   Answer: {answer}\n\n"
    
    return formatted_text

class SearchAPIClient:
    """Client for handling search API calls with session pooling."""
    
    # Class-level session pool shared across all instances
    _session_pool = {}
    _session_lock = threading.Lock()

    @classmethod
    def _get_shared_session(cls, base_url: str) -> requests.Session:
        """Get or create a shared session for the given base URL"""
        with cls._session_lock:
            if base_url not in cls._session_pool:
                session = requests.Session()
                # Configure connection pooling
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=20,  # Number of connection pools
                    pool_maxsize=20,  # Max connections per pool
                    max_retries=0,  # We handle retries ourselves
                    pool_block=False,  # Don't block if pool is full
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                cls._session_pool[base_url] = session
                logger.info(f"Created shared session pool for {base_url}")
            return cls._session_pool[base_url]

    def __init__(self, search_url=None, topk=3, timeout=DEFAULT_TIMEOUT, log_requests=True):
        # Get search URL from environment variable or use default
        self.search_url = search_url or os.getenv("SEARCH_API_URL", "http://127.0.0.1:8000/retrieve")
        self.topk = topk
        self.timeout = timeout
        self.log_requests = log_requests

        # Extract base URL for session sharing
        parsed_url = urlparse(self.search_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Get shared session for this base URL
        self.session = self._get_shared_session(self.base_url)
        if self.log_requests:
            logger.info(f"SearchAPIClient initialized using shared session pool for {self.base_url}")

    def search(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """
        Perform web search using the configured API.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return (overrides topk if provided)
            
        Returns:
            List of search result dictionaries
        """
        if not query or not query.strip():
            return []

        query = query.strip()
        topk = max_results if max_results is not None else self.topk

        try:
            api_response, error_msg = call_search_api(
                retrieval_service_url=self.search_url,
                query=query,
                topk=topk,
                timeout=self.timeout,
                log_requests=self.log_requests,
                session=self.session,
            )

            if error_msg:
                logger.error(f"Search API error for query '{query}': {error_msg}")
                return []

            if not api_response:
                logger.warning(f"No response from search API for query '{query}'")
                return []

            # Parse the API response
            raw_results = api_response.get("result", [])
            if not raw_results:
                logger.info(f"No search results found for query '{query}'")
                return []

            # Convert API response to expected format
            search_results = []
            for retrieval in raw_results:
                if isinstance(retrieval, list):
                    for idx, doc_item in enumerate(retrieval):
                        if isinstance(doc_item, dict) and "document" in doc_item:
                            content = doc_item["document"].get("contents", "").strip()
                            title = doc_item["document"].get("title", f"Search Result {idx+1}")
                            url = doc_item["document"].get("url", "")
                            
                            search_results.append({
                                "title": title,
                                "snippet": content,
                                "content": content,
                                "url": url
                            })

            if self.log_requests:
                logger.info(f"Search API returned {len(search_results)} results for query '{query}'")

            return search_results

        except Exception as e:
            logger.error(f"Exception during search API call for query '{query}': {e}")
            return []


# Global search client instance
_search_client = None

def get_search_client() -> SearchAPIClient:
    """Get or create the global search client instance."""
    global _search_client
    if _search_client is None:
        _search_client = SearchAPIClient()
    return _search_client

def simulate_web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Perform web search using real search APIs.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of search result dictionaries
    """
    if not query or not query.strip():
        logger.warning("Empty or None query provided to simulate_web_search")
        return []

    # Try to use real search API first
    try:
        search_client = get_search_client()
        results = search_client.search(query, max_results=max_results)
        
        if results:
            logger.info(f"Real search API returned {len(results)} results for query: '{query}'")
            return results
        else:
            logger.warning(f"Real search API returned no results for query: '{query}', falling back to mock")
            
    except Exception as e:
        logger.warning(f"Real search API failed for query '{query}': {e}, falling back to mock")

    # Fallback to mock search if real API fails or returns no results
    logger.info(f"Using mock search results for query: '{query}'")
    mock_results = [
        {
            "title": f"Search result for '{query}' - Article 1",
            "snippet": f"This article discusses {query} and provides detailed information about the topic.",
            "url": f"https://example.com/article1?q={query.replace(' ', '+')}"
        },
        {
            "title": f"Wikipedia: {query}",
            "snippet": f"Wikipedia article about {query} with comprehensive coverage.",
            "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        },
        {
            "title": f"Recent news about {query}",
            "snippet": f"Latest updates and news related to {query}.",
            "url": f"https://news.example.com/{query.replace(' ', '-')}"
        }
    ]
    
    return mock_results[:max_results]