import pandas as pd
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score
from utils import *
from display import *
from net import *

def data_preprocessing(corpus):
    """
    Present only for MSP-Podcast Corpus
    Wait for update
    """
    # Load corpus data
    corpus_path = config[corpus]["PATH_TO_LABEL"]
    corpus_df = pd.read_csv(corpus_path)
    corpus_df["FileName"]= corpus_df["FileName"].str.replace('.wav', '')

    # Remove non consensus labels
    main_corpus_df = corpus_df[~corpus_df["EmoClass"].isin(["X", "O"])]

    return main_corpus_df

def trainer(model,train_loader,val_loader,epochs,batch_size,learning_rate,use_feature,length, total_steps,patience):
    sample_per_class = [6716,6400,16652,2992,1134,1419,2519,29144]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = BalancedSoftmaxLoss(sample_per_class)
    lr_scheduler = get_scheduler(
        name="linear",                 
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=total_steps
    )
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
        train_length = length["train"]
        model.train()

        for batch_idx, (data, label) in enumerate(train_loader):
            
            category, avd = label["category"], label["avd"] # avd not use in current model
            true_label = torch.argmax(category, dim=1).cuda()

            mem_preforward = torch.cuda.memory_allocated()
            if use_feature:
                text, audio = data["text"], data["audio"]
                text, audio = text.cuda(), audio.cuda()
                output = model(audio,text)
            else:
                sample, sr = data["sample"], data["sr"]
                output = model(sample)
            mem_forward = torch.cuda.memory_allocated()
            
            loss = criterion(output,true_label)
            optimizer.zero_grad()
            loss.backward()
            mem_backward = torch.cuda.memory_allocated()
            optimizer.step()
            mem_optimizer = torch.cuda.memory_allocated()
            lr_scheduler.step()

            mem_metrics = {
                "mem/forward": mem_forward,
                "mem/consumed_by_forward": mem_forward - mem_preforward,
                "mem/backward": mem_backward,
                "mem/optimizer": mem_optimizer
            }
            wandb.log(mem_metrics)

            train_loss += loss.item()
            # For F1 score, accuracy, and UAR calculation
            pred = output.argmax(dim=1).cpu()
            true_label = torch.argmax(category, dim=1).cpu()
            running_macro_f1 += f1_score(true_label, pred, average='macro')
            running_acc += accuracy_score(true_label, pred)
            running_uar += recall_score(true_label, pred, average='macro',zero_division=0.0)


            progress_bar(batch_idx+1, train_length, 'Loss: %.3f | Macro F1: %.3f | Acc: %.3f | UAR: %.3f' % 
                        (train_loss/(batch_idx+1), running_macro_f1/(batch_idx+1), 
                        running_acc/(batch_idx+1), running_uar/(batch_idx+1)))
        
        avg_loss = train_loss / (batch_idx + 1)
        avg_macro_f1 = running_macro_f1 / (batch_idx + 1) 
        avg_acc = running_acc / (batch_idx + 1)
        avg_uar = running_uar / (batch_idx + 1)

        train_metric = {
            "train/loss": avg_loss,
            "train/macro_f1": avg_macro_f1,
            "train/accuracy": avg_acc,
            "train/uar": avg_uar,
        }

        running_macro_f1 = 0
        running_acc = 0
        running_uar = 0
        val_length = length["val"]
        model.eval()

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(val_loader):
                
                category, avd = label["category"], label["avd"] # avd not use in current model
                true_label = torch.argmax(category, dim=1).cuda()

                if use_feature:
                    text, audio = data["text"], data["audio"]
                    text, audio = text.cuda(), audio.cuda()
                    output = model(audio,text)
                else:
                    sample, sr = data["sample"], data["sr"]
                    output = model(sample)

                loss = criterion(output,true_label)
                val_loss += loss.item()
                
                pred = output.argmax(dim=1).cpu()
                true_label = torch.argmax(category, dim=1).cpu()
                running_macro_f1 += f1_score(true_label, pred, average='macro')
                running_acc += accuracy_score(true_label, pred)
                running_uar += recall_score(true_label, pred, average='macro',zero_division=0.0)

                if(batch_idx in batch_list):
                    log_view_table(sample,sr[0],pred,true_label,avd.cpu(),output.softmax(dim=1).cpu())
                

                progress_bar(batch_idx+1, val_length, 'Loss: %.3f | Macro F1: %.3f | Acc: %.3f | UAR: %.3f' % 
                            (val_loss/(batch_idx+1), running_macro_f1/(batch_idx+1), 
                            running_acc/(batch_idx+1), running_uar/(batch_idx+1)))
            
        avg_loss = val_loss / (batch_idx + 1)
        avg_macro_f1 = running_macro_f1 / (batch_idx + 1) 
        avg_acc = running_acc / (batch_idx + 1)
        avg_uar = running_uar / (batch_idx + 1)
        
        val_metric = {
            "val/loss": avg_loss,
            "val/macro_f1": avg_macro_f1,
            "val/accuracy": avg_acc,
            "val/uar": avg_uar,
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
    batch_size = kwargs.get('batch_size',16)
    epochs = kwargs.get('epoch',5)
    learning_rate = kwargs.get('learning_rate',5e-5)
    verbose = kwargs.get('verbose',True)
    patience = kwargs.get('patience',5)
    save_path = kwargs.get('save_path', 'saved_models')

    torch.cuda.empty_cache()

    wandb.login(key=config['WANDBAPI'])
    wandb.init(
        project="MSP-Podcast",
        tags=["baseline","Finetune try"], 
        config = {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "use_feature": use_feature,
            "dropout": kwargs.get('dropout',0.2),
            "model_type": model_type,
            "upstream_model": kwargs.get('upstream_model',"wav2vec2-large-960h"),
            "finetune_layers": kwargs.get('finetune_layers',1),
            "hidden_dim": kwargs.get('hidden_dim',64),
            "num_layers": kwargs.get('num_layers',42),
        }
    )
    
    wandb.log({"mem/begin": torch.cuda.memory_allocated()})

    if model_type == 'UpstreamFinetune':
        model = Upstream_finetune_simple(
            upstream_name=kwargs.get('upstream_model', "wav2vec2-large-960h"),
            finetune_layers=kwargs.get('finetune_layers', 2),
            hidden_dim=kwargs.get('hidden_dim', 64),
            dropout=kwargs.get('dropout', 0.2),
            num_layers=kwargs.get('num_layers', 2),
            device=device
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    wandb.log({"mem/model_begin": torch.cuda.memory_allocated()})
    
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Dataset Preparation
    corpus_dataFrame = data_preprocessing(corpus)
    train_dataFrame, val_dataFrame = corpus_split(corpus_dataFrame)
    train_dataset = MSPDataset(train_dataFrame,audio_path=config[corpus]['PATH_TO_AUDIO'],use_feature=use_feature,seed=seed)
    val_dataset = MSPDataset(val_dataFrame,audio_path=config[corpus]['PATH_TO_AUDIO'],use_feature=use_feature,seed=seed)

    if verbose:
        print(f"Number of training samples: {train_dataset.total_samples}")
        print(f"Number of validation samples: {val_dataset.total_samples}")
        
        numel_list = [p.numel() for p in model.parameters() if p.requires_grad]
        total_params = sum(numel_list)
        print(f"Total number of trainable parameters: {total_params:,}")
        wandb.log({"model/trainable_parameters": total_params})

    length = {
        "train": train_dataset.total_samples // batch_size + 1,
        "val": val_dataset.total_samples // batch_size + 1
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size*torch.cuda.device_count(), 
                            num_workers=4*torch.cuda.device_count(), pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*torch.cuda.device_count(), 
                            num_workers=4*torch.cuda.device_count(), pin_memory=True, collate_fn=collate_fn)
    total_training_steps = train_dataset.total_samples // batch_size * epochs

    model, results = trainer(model,train_loader,val_loader,epochs,batch_size,learning_rate,use_feature,length, total_training_steps, patience)

    wandb.finish()
    
    # Save the best model
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(save_path, f"best_model_{timestamp}.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict().copy(),
        'val_loss': results['best_val_loss'],
        'config': wandb.config
    }, best_model_path)
    print(f"Best model saved to {best_model_path}")