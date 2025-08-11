"""
Environment for web design tasks that handles both code generation and visual analysis.
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING
import shutil
from datasets import load_dataset  # type: ignore
from tqdm import tqdm
from pettingllms.environments.base.multi_turn_env import MultiTurnEnvironment
from web_utils import render_full_html, get_driver, replace_urls
import re
import time

if TYPE_CHECKING:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from torch import Tensor
    from PIL.Image import Image as PILImage

try:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    print("⚠️ Some required packages are missing. Please run 'pip install torch Pillow transformers'")
    torch = None
    Image = None
    CLIPModel = None
    CLIPProcessor = None



class WebEnv(MultiTurnEnvironment):
    """
    Environment for web design tasks with dual-agent interaction.
    """

    def __init__(self, task: Optional[Dict] = None, max_turns: int = 3, temp_path: Optional[str] = "temp/"+str(time.time())):
        """
        Initialize the web design environment.

        Args:
            task: Dictionary containing the task information
            max_turns: Maximum number of turns before terminating
            temp_path: Path for temporary files
        """
        super().__init__(task=task, max_turns=max_turns)
        self.temp_path = temp_path
        self.current_html: Optional[str] = None
        self.driver = None
        self.reference_image: Optional[str] = None
        self.current_image: Optional[str] = None
        self.clip_model: Optional[Any] = None
        self.clip_processor: Optional[Any] = None
        # HTML file paths for result storage
        self.reference_html_path: Optional[str] = None
        self.current_html_path: Optional[str] = None
        
        # Initialize CLIP model and processor
        if torch is not None and CLIPModel is not None and CLIPProcessor is not None:
            try:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                print("✅ CLIP model initialized successfully")
            except Exception as e:
                print(f"⚠️ CLIP model initialization failed: {e}")
                self.clip_model = None
                self.clip_processor = None
        else:
            print("⚠️ CLIP dependencies not available")
            self.clip_model = None
            self.clip_processor = None
        
        # Initialize webdriver
        print("🔄 Initializing WebDriver...")
        self.driver = get_driver()
        
        if self.driver:
            # Test WebDriver functionality
            try:
                self.driver.get("data:text/html,<html><body><h1>Test</h1></body></html>")
                print("✅ WebDriver test page loaded successfully")
                self.driver_available = True
            except Exception as test_e:
                print(f"⚠️ WebDriver test failed: {test_e}")
                print("🔧 Switching to mock mode")
                try:
                    self.driver.quit()
                except:
                    pass
                self.driver = None
                self.driver_available = False
        else:
            print("🔧 No WebDriver available, using mock mode")
            self.driver_available = False
              # Render ground truth image
        if self.driver:
            try:
                result = render_full_html(
                    self.driver,
                    self.task["ground_truth"],
                    self.temp_path+"/gt",
                    env_id=self.task["task_id"]  # type: ignore
                )
                if result and result[0]:
                    self.reference_image = result[0]
                    self.reference_html_path = result[1]
                else:
                    self.reference_image = None
                    self.reference_html_path = None
            except Exception as e:
                print(f"❌ Ground truth rendering failed: {e}")
                self.reference_image = None
                self.reference_html_path = None
        else:
            # Mock mode for ground truth
            try:
                import time
                current_time = time.time()
                mock_gt_html_path = os.path.join(self.temp_path, f"{self.task['task_id']}_gt_{current_time}.html")
                mock_gt_image_path = os.path.join(self.temp_path, f"{self.task['task_id']}_gt_{current_time}.png")
                
                # Save ground truth HTML
                with open(mock_gt_html_path, 'w', encoding='utf-8') as f:
                    f.write(self.task["ground_truth"])
                print(f"💾 Ground truth HTML saved to: {mock_gt_html_path}")
                
                # Create mock ground truth image
                import base64
                black_pixel_png = base64.b64decode(
                    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
                )
                with open(mock_gt_image_path, 'wb') as f:
                    f.write(black_pixel_png)
                print(f"🖼️ Mock ground truth image created: {mock_gt_image_path}")
                
                self.reference_image = mock_gt_image_path
                self.reference_html_path = mock_gt_html_path
                
            except Exception as mock_e:
                print(f"❌ Mock ground truth creation failed: {mock_e}")
                self.reference_image = None
                self.reference_html_path = None


    def get_reward_and_next_obs(self, task: Dict, action: str) -> Tuple[float, Dict]:
        """
        Compute reward and next observation. This is a placeholder implementation
        as the actual reward is handled in the step function.
        
        Args:
            task: Task dictionary
            action: Action string
            
        Returns:
            Tuple of (reward, next_observation)
        """
        return 0.0, {}

    def reset(self, task=None, seed=None):
        """Reset the environment and return initial observations."""
        import random
        import os

        if seed is not None:
            random.seed(seed)

        if task is not None:
            self.task = task

        assert self.task is not None, "Task must be set before reset"

        self.done = False
        self.current_turn = 0
        self.history = []
        self.current_html = None
        
      
        return {
            "task_description": self.task["problem_description"],
            "reference_image": self.reference_image
        }, {}

    def render_html(self, html_code: str, iteration: int = 0) -> Optional[str]:
        """
        Render HTML code to image
        
        Args:
            html_code: HTML code to render
            iteration: Current iteration number
            
        Returns:
            Path to rendered image file
        """
        if not self.driver:
            print("⚠️ WebDriver not available, using mock mode")
            print("   💡 HTML will be saved but no image rendering")
            # Create mock mode - save HTML but return mock image path
            try:
                import time
                current_time = time.time()
                mock_html_path = os.path.join(self.temp_path, f"{self.task['task_id']}_{iteration}_{current_time}.html")
                mock_image_path = os.path.join(self.temp_path, f"{self.task['task_id']}_{iteration}_{current_time}.png")
                
                # Save HTML file
                with open(mock_html_path, 'w', encoding='utf-8') as f:
                    f.write(html_code)
                print(f"💾 HTML saved to: {mock_html_path}")
                
                # Create a simple placeholder image (1x1 pixel black PNG)
                import base64
                # Minimal 1x1 black PNG image in base64
                black_pixel_png = base64.b64decode(
                    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
                )
                with open(mock_image_path, 'wb') as f:
                    f.write(black_pixel_png)
                print(f"🖼️ Mock image created: {mock_image_path}")
                
                self.current_html_path = mock_html_path
                return mock_image_path
                
            except Exception as mock_e:
                print(f"❌ Mock mode failed: {mock_e}")
                return None
            
        if not html_code or html_code.strip() == "":
            print("❌ Rendering failed: Empty HTML code provided")
            return None
            
        if not self.temp_path:
            print("❌ Rendering failed: No temp_path specified")
            return None
            
        # Basic HTML validation
        html_lower = html_code.lower().strip()
        if not (html_lower.startswith('<html') or html_lower.startswith('<!doctype')):
            print("⚠️ Warning: HTML doesn't start with <html> or <!doctype>")
            print(f"   📋 HTML starts with: {html_code[:50]}...")
            
        if '</html>' not in html_lower:
            print("⚠️ Warning: HTML doesn't contain closing </html> tag")
            
        # Check if temp_path directory exists and is writable
        try:
            if not os.path.exists(self.temp_path):
                os.makedirs(self.temp_path, exist_ok=True)
                print(f"📁 Created temp directory: {self.temp_path}")
            
            # Test write permissions
            test_file = os.path.join(self.temp_path, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"✅ Temp directory is writable: {self.temp_path}")
            
        except Exception as dir_e:
            print(f"❌ Temp directory issue: {dir_e}")
            return None
            
        try:
            print(f"🔄 Attempting to render HTML (iteration {iteration})...")
            print(f"   📝 HTML length: {len(html_code)} characters")
            print(f"   📁 Temp path: {self.temp_path}")
            
            result = render_full_html(
                self.driver,
                html_code,
                self.temp_path,
                env_id=f"{self.task['task_id']}_{iteration}"  # type: ignore
            )
            
            if result and result[0] and os.path.exists(result[0]):
                image_path, html_path = result
                print(f"✅ Successfully rendered to: {image_path}")
                print(f"💾 HTML saved to: {html_path}")
                # Store HTML path for later use
                self.current_html_path = html_path
                return image_path
            else:
                print("❌ Rendering failed: render_full_html returned None or file doesn't exist")
                if result and result[0]:
                    print(f"   🔍 Expected image path: {result[0]}")
                    print(f"   📂 Image path exists: {os.path.exists(result[0])}")
                self.current_html_path = None
                return None
                
        except Exception as e:
            import traceback
            print(f"❌ Rendering failed with exception: {type(e).__name__}: {e}")
            print(f"   📋 Full traceback:")
            traceback.print_exc()
            
            # Additional diagnostic information
            print(f"   🔍 Diagnostic info:")
            print(f"      - Driver status: {'Available' if self.driver else 'None'}")
            print(f"      - Task ID: {getattr(self.task, 'task_id', 'Unknown') if self.task else 'No task'}")
            print(f"      - HTML preview: {html_code[:200]}..." if len(html_code) > 200 else f"      - HTML content: {html_code}")
            
            return None

    def step(self, role: str, action: str):
        """
        Take a step in the environment based on the role and action.

        Args:
            role: Either 'code' or 'visual' indicating the agent role
            action: Response string from the agent

        Returns:
            next_observation, reward, terminated, truncated, info
        """
        # Store the action in history
        self.history.append({"role": role, "action": action})

        # Handle code generation role
        if role == "code":
            self.current_html = action
            self.current_image = self.render_html(action, self.current_turn)
            
            # Calculate CLIP-based reward if both images are available
            reward = 0.0
            if (self.current_image and self.reference_image and 
                self.clip_model is not None and self.clip_processor is not None and 
                Image is not None and torch is not None):
                try:
                    # Load and process images
                    current_img = Image.open(self.current_image).convert("RGB")
                    reference_img = Image.open(self.reference_image).convert("RGB")
                    
                    # Process images through CLIP
                    inputs1 = self.clip_processor(images=current_img, return_tensors="pt", padding=True).to("cuda")
                    inputs2 = self.clip_processor(images=reference_img, return_tensors="pt", padding=True).to("cuda")
                    
                    # Get image embeddings
                    with torch.no_grad():  # type: ignore
                        image_features1 = self.clip_model.get_image_features(**inputs1)
                        image_features2 = self.clip_model.get_image_features(**inputs2)
                    
                    # Normalize embeddings
                    image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)
                    image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)
                    
                    # Calculate cosine similarity as reward
                    reward = float(torch.sum(image_features1 * image_features2, dim=-1).cpu().numpy())  # type: ignore
                except Exception as e:
                    print(f"⚠️ Error calculating CLIP reward: {e}")
                    reward = 0.0
            
            next_obs = {
                "reference_image": self.reference_image,
                "current_image": self.current_image,
                "current_html_path": self.current_html_path
            }
            
        # Handle visual analysis role
        elif role == "visual":
            suggestions = action
            next_obs = {
                "task_description": self.task["problem_description"],  # type: ignore
                "visual_suggestions": suggestions
            }
            reward = 0.0  # Visual analysis role doesn't get immediate reward
            
        else:
            raise ValueError(f"Invalid role: {role}")

        # Increment turn counter
        self.current_turn += 1
        
        # Check if we've reached the maximum number of turns
        if self.current_turn >= self.max_turns:
            self.done = True

        return next_obs, reward, self.done, next_obs

    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            try:
                self.driver.quit()
                print("✅ WebDriver closed successfully")
            except Exception as e:
                print(f"⚠️ Error closing WebDriver: {e}")

    @staticmethod
    def from_dict(env_args: dict) -> "WebEnv":
        return WebEnv(
            task=env_args["task"],
            max_turns=env_args.get("max_turns", 3),
            temp_path=env_args.get("temp_path")
        )
