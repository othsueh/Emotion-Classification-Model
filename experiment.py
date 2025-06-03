import pandas as pd
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score
from transformers import get_scheduler
from utils import *
from display import *
from net import *
from dataset_module import *

sr = 16000

def trainer(model,dataset,train_loader,val_loader,epochs,batch_size,learning_rate,use_feature,use_gender,length, total_steps,patience):
    sample_per_class = dataset.sample_per_class
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    category_criterion = BalancedSoftmaxLoss(sample_per_class)
    dim_criterion = CCCLoss()
    # Scheduler commented out but kept for reference
    # lr_scheduler = get_scheduler(
    #     name="linear",                 
    #     optimizer=optimizer,
    #     num_warmup_steps=500,
    #     num_training_steps=total_steps
    # )
    batch_list = [0,1,2,3]

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for ep in range(epochs):
        print('='*30+f'Epoch {ep+1}/{epochs}'+'='*30)

        train_loss = 0
        val_loss = 0
        running_macro_f1 = 0
        running_acc = 0
        running_uar = 0
        running_ccc_arousal = 0
        running_ccc_valence = 0
        train_length = length["train"]
        model.train()

        for batch_idx, data in enumerate(train_loader):
            
            category, av = data["category"], data["av"] # av not use in current model
            true_label = torch.argmax(category, dim=1).cuda()
            av = av.cuda()

            # Calculate preforward memory usage
            mem_preforward = torch.cuda.memory_allocated()

            audio = data["audio"]
            gender = data["gender"]

            
            if use_gender == True:
                gender = gender.cuda()
                category_output, dim_output = model(audio,sr,gender)
            else:
                category_output, dim_output = model(audio,sr)

            # Calculate forwarded memory usage
            mem_forward = torch.cuda.memory_allocated()
            
            cls_loss = category_criterion(category_output,true_label)
            # !Debug
            reg_loss, ccc_arousal, ccc_valence = dim_criterion(dim_output,av)
            alpha,beta = 1, 1
            total_loss = alpha * reg_loss + beta * cls_loss
            optimizer.zero_grad()
            total_loss.backward()
            mem_backward = torch.cuda.memory_allocated()
            optimizer.step()
            mem_optimizer = torch.cuda.memory_allocated()
            # Scheduler step commented out
            # lr_scheduler.step()
            
            mem_metrics = {
                "mem/forward": mem_forward,
                "mem/consumed_by_forward": mem_forward - mem_preforward,
                "mem/backward": mem_backward,
                "mem/optimizer": mem_optimizer
            }
            wandb.log(mem_metrics)

            train_loss += total_loss.item()
            # For F1 score, accuracy, and UAR calculation
            pred = category_output.argmax(dim=1).cpu()
            true_label = torch.argmax(category, dim=1).cpu()
            running_macro_f1 += f1_score(true_label, pred, average='macro')
            running_acc += accuracy_score(true_label, pred)
            running_uar += recall_score(true_label, pred, average='macro',zero_division=0.0)
            running_ccc_arousal += ccc_arousal.item()
            running_ccc_valence += ccc_valence.item()

            torch.cuda.empty_cache()


            progress_bar(batch_idx+1, train_length, 'Loss: %.3f | Macro F1: %.3f | Acc: %.3f | UAR: %.3f | Arousal: %.3f | Valence: %.3f' % 
                        (train_loss/(batch_idx+1), running_macro_f1/(batch_idx+1), 
                        running_acc/(batch_idx+1), running_uar/(batch_idx+1),
                        running_ccc_arousal/(batch_idx+1), running_ccc_valence/(batch_idx+1)))
        
        avg_loss = train_loss / (batch_idx + 1)
        avg_macro_f1 = running_macro_f1 / (batch_idx + 1) 
        avg_acc = running_acc / (batch_idx + 1)
        avg_uar = running_uar / (batch_idx + 1)
        avg_arousal = running_ccc_arousal / (batch_idx + 1)
        avg_valence = running_ccc_valence / (batch_idx + 1)
        avg_ccc = (avg_arousal + avg_valence) / 2

        train_metric = {
            "train/loss": avg_loss,
            "train/macro_f1": avg_macro_f1,
            "train/accuracy": avg_acc,
            "train/uar": avg_uar,
            "train/arousal": avg_arousal,
            "train/valence": avg_valence,
            "train/avg_ccc": avg_ccc,
        }

        running_macro_f1 = 0
        running_acc = 0
        running_uar = 0
        running_ccc_arousal = 0
        running_ccc_valence = 0
        val_length = length["val"]
        model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                
                category, av = data["category"], data["av"] # av not use in current model
                true_label = torch.argmax(category, dim=1).cuda()
                av = av.cuda()

                audio = data["audio"]
                gender = data["gender"]
                audio = audio.squeeze(1).cuda()

                if use_gender == True:
                    gender = gender.cuda()
                    category_output, dim_output = model(audio,sr,gender)
                else:
                    category_output, dim_output = model(audio,sr)

                cls_loss = category_criterion(category_output,true_label)
                reg_loss, ccc_arousal, ccc_valence = dim_criterion(dim_output,av)
                total_loss = alpha * reg_loss + beta * cls_loss
                val_loss += total_loss.item()
                
                pred = category_output.argmax(dim=1).cpu()
                true_label = torch.argmax(category, dim=1).cpu()
                running_macro_f1 += f1_score(true_label, pred, average='macro')
                running_acc += accuracy_score(true_label, pred)
                running_uar += recall_score(true_label, pred, average='macro',zero_division=0.0)
                running_ccc_arousal += ccc_arousal
                running_ccc_valence += ccc_valence

                if(batch_idx in batch_list):
                    arousal = dim_output[:,0].cpu()
                    valence = dim_output[:,1].cpu()
                    log_view_table(dataset,audio.cpu().numpy(),sr,pred,true_label,arousal,valence,av.cpu(),category_output.softmax(dim=1).cpu())
                

                progress_bar(batch_idx+1, val_length, 'Loss: %.3f | Macro F1: %.3f | Acc: %.3f | UAR: %.3f | Arousal: %.3f | Valence: %.3f' % 
                        (val_loss/(batch_idx+1), running_macro_f1/(batch_idx+1), 
                        running_acc/(batch_idx+1), running_uar/(batch_idx+1),
                        running_ccc_arousal/(batch_idx+1), running_ccc_valence/(batch_idx+1)))
            
        avg_loss = val_loss / (batch_idx + 1)
        avg_macro_f1 = running_macro_f1 / (batch_idx + 1) 
        avg_acc = running_acc / (batch_idx + 1)
        avg_uar = running_uar / (batch_idx + 1)
        avg_arousal = running_ccc_arousal / (batch_idx + 1)
        avg_valence = running_ccc_valence / (batch_idx + 1)
        avg_ccc = (avg_arousal + avg_valence) / 2
        
        val_metric = {
            "val/loss": avg_loss,
            "val/macro_f1": avg_macro_f1,
            "val/accuracy": avg_acc,
            "val/uar": avg_uar,
            "val/arousal": avg_arousal,
            "val/valence": avg_valence,
            "val/avg_ccc": avg_ccc,
        }

        # Early stopping check
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epochs")
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {ep+1} epochs")
            model.load_state_dict(best_model_state)  # Restore best model
            wandb.log({**train_metric,**val_metric})
            break
        wandb.log({**train_metric,**val_metric})

    return model, {"best_val_loss": best_val_loss}




def run_experiment(model_type,device='cuda',**kwargs):
    corpus = kwargs.get('corpus','MSPPODCAST')
    seed = kwargs.get('seed',42)  
    use_feature = kwargs.get('use_feature',False)  
    use_gender = kwargs.get('use_gender',False)  
    batch_size = kwargs.get('batch_size',16)
    epochs = kwargs.get('epoch',5)
    learning_rate = kwargs.get('learning_rate',5e-5)
    verbose = kwargs.get('verbose',True)
    patience = kwargs.get('patience',5)

    torch.cuda.empty_cache()

    wandb.login(key=WANDB_TOKEN)
    run = wandb.init(
        project="Total Set",
        tags=["Small set","Gender Tune","Dual Head"], 
        config = {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "use_feature": use_feature,
            "use_gender": use_gender,
            "dropout": kwargs.get('dropout',0.2),
            "model_type": model_type,
            "upstream_model": kwargs.get('upstream_model',"wav2vec2-large-960h"),
            "finetune_layers": kwargs.get('finetune_layers',1),
            "hidden_dim": kwargs.get('hidden_dim',64),
            "num_layers": kwargs.get('num_layers',2),
            "classifier_output_dim": kwargs.get('classifier_output_dim', 8),
        }
    )

    wandb.log({"mem/begin": torch.cuda.memory_allocated()})

    if model_type == 'UpstreamFinetune':
        model_config = UpstreamFinetuneConfig(
            origin_upstream_url=kwargs.get('origin_upstream_url',"facebook/wav2vec2-base-960h"),
            upstream_model=kwargs.get('upstream_model', "wav2vec2-base-960h"),
            finetune_layers=kwargs.get('finetune_layers', 2),
            hidden_dim=kwargs.get('hidden_dim', 64),
            dropout=kwargs.get('dropout', 0.2),
            num_layers=kwargs.get('num_layers', 2),
            classifier_output_dim=kwargs.get('classifier_output_dim', 8),
        )
        model = UpstreamFinetune(model_config,config["PATH_TO_PRETRAINED_MODELS"],device)
    elif model_type == 'UpstreamTest1':
        model_config = UpstreamFinetuneConfig(
            origin_upstream_url=kwargs.get('origin_upstream_url',"facebook/wav2vec2-base-960h"),
            upstream_model=kwargs.get('upstream_model', "wav2vec2-base-960h"),
            finetune_layers=kwargs.get('finetune_layers', 2),
            hidden_dim=kwargs.get('hidden_dim', 64),
            dropout=kwargs.get('dropout', 0.2),
            num_layers=kwargs.get('num_layers', 2),
            classifier_output_dim=kwargs.get('classifier_output_dim', 8),
        )
        model = UpstreamTest1(model_config,config["PATH_TO_PRETRAINED_MODELS"],device)
    elif model_type == 'UpstreamTest2':
        model_config = UpstreamFinetuneConfig(
            origin_upstream_url=kwargs.get('origin_upstream_url',"facebook/wav2vec2-base-960h"),
            upstream_model=kwargs.get('upstream_model', "wav2vec2-base-960h"),
            finetune_layers=kwargs.get('finetune_layers', 2),
            hidden_dim=kwargs.get('hidden_dim', 64),
            dropout=kwargs.get('dropout', 0.2),
            num_layers=kwargs.get('num_layers', 2),
            classifier_output_dim=kwargs.get('classifier_output_dim', 8),
        )
        model = UpstreamTest2(model_config,config["PATH_TO_PRETRAINED_MODELS"],device)
    elif model_type == 'UpstreamTest3':
        model_config = UpstreamFinetuneConfig(
            origin_upstream_url=kwargs.get('origin_upstream_url',"facebook/wav2vec2-base-960h"),
            upstream_model=kwargs.get('upstream_model', "wav2vec2-base-960h"),
            finetune_layers=kwargs.get('finetune_layers', 2),
            hidden_dim=kwargs.get('hidden_dim', 64),
            dropout=kwargs.get('dropout', 0.2),
            num_layers=kwargs.get('num_layers', 2),
            classifier_output_dim=kwargs.get('classifier_output_dim', 8),
        )
        model = UpstreamTest3(model_config,config["PATH_TO_PRETRAINED_MODELS"],device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    wandb.log({"mem/model_begin": torch.cuda.memory_allocated()})
    
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Dataset Preparation (Just need to load webdataset)
    dataset = CombineCorpus(config[corpus]['PATH_TO_DATASET'])
    train_samples = dataset.train_counts
    val_samples = dataset.validation_counts
    train_loader = dataset.create_dataloader('train',batch_size)
    valid_loader = dataset.create_dataloader('validation',batch_size)

    if verbose:
        print(f'There are total {train_samples} train samples')
        print(f'There are total {val_samples} validation samples')
        numel_list = [p.numel() for p in model.parameters() if p.requires_grad]
        total_params = sum(numel_list)
        print(f"Total number of trainable parameters: {total_params:,}")
        wandb.log({"model/trainable_parameters": total_params})

    length = {
        "train": train_samples // batch_size + 1,
        "val": val_samples // batch_size + 1
    }

    total_training_steps = train_samples // batch_size * epochs

    model, results = trainer(model,dataset,train_loader,valid_loader,epochs,batch_size,learning_rate,use_feature,use_gender,length, total_training_steps, patience)

    wandb.finish()
    
    # Save the best model
    save_path = config["PATH_TO_SAVED_MODELS"]
    model_path = os.path.join(save_path,run.name)
    model.save_pretrained(model_path)
    print(f"Best model saved to {model_path}")
