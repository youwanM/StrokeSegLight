import os

# 1. Manually set your ENV variables
os.environ['nnUNet_raw'] = "/home/ymahe/Desktop/StrokeSegLight/Distillation/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "/home/ymahe/Desktop/StrokeSegLight/Distillation/nnUNet_preprocessed"
os.environ['nnUNet_results'] = "/home/ymahe/Desktop/StrokeSegLight/Distillation/nnUNet_results"

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from batchgenerators.utilities.file_and_folder_operations import load_json, join

# --- IMPORT ALL STUDENT SIZES ---
from models import get_extra_extralight_student, get_extra_light_student, get_light_student, get_small_student, get_medium_student, get_large_student


# =========================================================================
# BASE TRAINER: Handles KD Logic, Loss, and Teacher Loading
# =========================================================================
class nnUNetTrainer_KD_Base(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # KD Hyperparameters
        self.temperature = 4.0
        self.alpha = 0.5  # Equal weight to Hard (Ground Truth) and Soft (Teacher) loss
        self.num_epochs = 1000
        
        # Fixed path to match standard nnU-Net folder and file naming
        self.teacher_weights_path = join(nnUNet_results, "Dataset999", "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres", "fold_2", "checkpoint_best.pth")
        self.teacher_network = None

    def on_train_start(self):
        super().on_train_start()
        
        # Build the TEACHER network using nnU-Net's utility function
        self.print_to_log_file("Loading Teacher Model...")
        self.teacher_network = get_network_from_plans(
            self.configuration_manager.network_arch_class_name,
            self.configuration_manager.network_arch_init_kwargs,
            self.configuration_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels,
            self.label_manager.num_segmentation_heads,
            allow_init=True,
            deep_supervision=self.enable_deep_supervision
        ).to(self.device)
        
        # Load Teacher weights
        if not os.path.isfile(self.teacher_weights_path):
            raise FileNotFoundError(f"Teacher weights not found at: {self.teacher_weights_path}")
            
        checkpoint = torch.load(self.teacher_weights_path, map_location=self.device, weights_only=False)
        self.teacher_network.load_state_dict(checkpoint['network_weights'])
        
        # Freeze Teacher
        for param in self.teacher_network.parameters():
            param.requires_grad = False
        self.teacher_network.eval()
        self.print_to_log_file("Teacher Model loaded and frozen.")

    def kd_loss_fn(self, student_logits, teacher_logits):
            """Calculates Kullback-Leibler divergence between softened logits."""
            # 1. Cast to FP32 to prevent FP16 overflow/NaNs during exponentiation
            student_logits = student_logits.float()
            teacher_logits = teacher_logits.float()
            
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
            
            # 2. Use reduction='none' to prevent summing over 2 million voxels
            soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='none')
            
            # Sum over the class dimension (dim=1), then average over batch and spatial dimensions
            soft_loss = soft_loss.sum(dim=1).mean()
            
            return soft_loss * (self.temperature ** 2)

    def train_step(self, batch: dict) -> dict:
        """
        Override the training step. Matches the new `batch: dict` input requirement.
        """
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        
        # Target handling to match the parent class
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            student_output = self.network(data)
            
            with torch.no_grad():
                self.teacher_network.eval() 
                teacher_output = self.teacher_network(data)
                
            hard_loss = self.loss(student_output, target)
            
            if self.enable_deep_supervision:
                kd_loss = self.kd_loss_fn(student_output[0], teacher_output[0])
            else:
                kd_loss = self.kd_loss_fn(student_output, teacher_output)

            total_loss = (1. - self.alpha) * hard_loss + self.alpha * kd_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss.detach().cpu().numpy()}


# =========================================================================
# SUBCLASSES: Define the specific network size to train
# =========================================================================

class nnUNetTrainer_KD_Large(nnUNetTrainer_KD_Base):
    def build_network_architecture(self, plans_manager, configuration_manager, num_input_channels, num_output_channels, enable_deep_supervision=True):
        return get_large_student()
    
class nnUNetTrainer_KD_Medium(nnUNetTrainer_KD_Base):
    def build_network_architecture(self, plans_manager, configuration_manager, num_input_channels, num_output_channels, enable_deep_supervision=True):
        return get_medium_student()

class nnUNetTrainer_KD_Small(nnUNetTrainer_KD_Base):
    def build_network_architecture(self, plans_manager, configuration_manager, num_input_channels, num_output_channels, enable_deep_supervision=True):
        return get_small_student()

class nnUNetTrainer_KD_Light(nnUNetTrainer_KD_Base):
    def build_network_architecture(self, plans_manager, configuration_manager, num_input_channels, num_output_channels, enable_deep_supervision=True):
        return get_light_student()

class nnUNetTrainer_KD_ExtraLight(nnUNetTrainer_KD_Base):
    def build_network_architecture(self, plans_manager, configuration_manager, num_input_channels, num_output_channels, enable_deep_supervision=True):
        return get_extra_light_student()

class nnUNetTrainer_KD_ExtraExtraLight(nnUNetTrainer_KD_Base):
    def build_network_architecture(self, plans_manager, configuration_manager, num_input_channels, num_output_channels, enable_deep_supervision=True):
        return get_extra_extralight_student()

# =========================================================================
# EXECUTION
# =========================================================================

def run_custom_kd_training(dataset_id: int, configuration: str, fold: int):
    dataset_name = maybe_convert_to_dataset_name(dataset_id)
    plans_file = join(nnUNet_preprocessed, dataset_name, 'nnUNetResEncUNetMPlans.json')
    plans = load_json(plans_file)
    
    # Inject the missing runtime flag that nnU-Net expects
    plans['continue_training'] = False 
    
    dataset_json = load_json(join(nnUNet_preprocessed, dataset_name, 'dataset.json'))
    
    trainer = nnUNetTrainer_KD_ExtraExtraLight( 
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=torch.device('cuda'),
    )
    
    trainer.initialize()
    trainer.save_every = 1
    checkpoint_path = join(trainer.output_folder, 'checkpoint_latest.pth')
    print("Number of trainable parameters in the student model:", sum(p.numel() for p in trainer.network.parameters() if p.requires_grad))
    
    if os.path.isfile(checkpoint_path):
        trainer.print_to_log_file(f"Found existing checkpoint! Resuming from: {checkpoint_path}")
        # This loads weights, optimizer, scheduler, and the current epoch number
        trainer.load_checkpoint(checkpoint_path)
    else:
        trainer.print_to_log_file("No checkpoint found. Starting training from scratch.")
    # ---------------------------        
    
    trainer.run_training()

if __name__ == '__main__':
    run_custom_kd_training(
        dataset_id=999, 
        configuration='3d_fullres', 
        fold=2
    )