from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from utils import *
from display import *
from net import *
from dataset_module import *

sr = 16000

def tester(model, test_loader, use_gender, use_dim_emotion, length, dataset):
    dim_criterion = CCCLoss()
    running_macro_f1 = 0
    running_acc = 0
    running_uar = 0
    if use_dim_emotion:
        running_ccc_arousal = 0
        running_ccc_valence = 0
    test_length = length
    model.eval()
    
    # Collect prediction data
    all_true_labels = []
    all_pred_labels = []
    all_genders = []
    if use_dim_emotion:
        all_true_av = []
        all_pred_av = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            
            category = data["category"] 
            true_label = torch.argmax(category, dim=1).cuda()

            if use_dim_emotion: 
                av = data["av"]
                all_true_av.extend(av.numpy())
                av = av.cuda()

            gender = data["gender"]
            gender_indices = torch.argmax(gender, dim=1).cpu().numpy()
            all_genders.extend(gender_indices)

            audio = data["audio"]
            audio = audio.squeeze(1).cuda()

            if use_gender:
                gender = gender.cuda()
                category_output, dim_output = model(audio, sr, gender)
            else:
                category_output, dim_output = model(audio, sr)

            pred = category_output.argmax(dim=1).cpu()
            true_label = torch.argmax(category, dim=1).cpu()
            
            # Collect emotion labels
            all_true_labels.extend(true_label.numpy())
            all_pred_labels.extend(pred.numpy())
            
            # Collect dimensional emotion predictions
            if use_dim_emotion:
                all_pred_av.extend(dim_output.cpu().numpy())
            
            running_macro_f1 += f1_score(true_label, pred, average='macro')
            running_acc += accuracy_score(true_label, pred)
            running_uar += recall_score(true_label, pred, average='macro', zero_division=0.0)

            if use_dim_emotion:
                reg_loss, ccc_arousal, ccc_valence = dim_criterion(dim_output, av)
                running_ccc_arousal += ccc_arousal
                running_ccc_valence += ccc_valence
                progress_bar(batch_idx+1, test_length, 'Macro F1: %.3f | Acc: %.3f | UAR: %.3f | Arousal: %.3f | Valence: %.3f' % 
                        (running_macro_f1/(batch_idx+1), running_acc/(batch_idx+1), 
                        running_uar/(batch_idx+1), running_ccc_arousal/(batch_idx+1), running_ccc_valence/(batch_idx+1)))
            else:
                progress_bar(batch_idx+1, test_length, 'Macro F1: %.3f | Acc: %.3f | UAR: %.3f' % 
                        (running_macro_f1/(batch_idx+1), running_acc/(batch_idx+1), 
                        running_uar/(batch_idx+1)))
            
            # Comment this for full evaluation
            # if batch_idx >= 2:  # Process a few batches for testing
            #    break
        
    avg_macro_f1 = f1_score(all_true_labels, all_pred_labels, average='macro',zero_division=0.0) 
    avg_acc = accuracy_score(all_true_labels, all_pred_labels)
    avg_uar = recall_score(all_true_labels, all_pred_labels, average='macro',zero_division=0.0)
    test_metric = {
        "test/macro_f1": avg_macro_f1,
        "test/accuracy": avg_acc,
        "test/uar": avg_uar,
    }
    separate_f1 = f1_score(all_true_labels, all_pred_labels, average=None,zero_division=0.0) 
    separate_recall = recall_score(all_true_labels, all_pred_labels, average=None,zero_division=0.0) 
    separate_precision = precision_score(all_true_labels, all_pred_labels, average=None,zero_division=0.0) 
    
    score_table = wandb.Table(columns=["Emotion", "F1", "Recall", "Precision"], data=[[emotion,f1,recall,precision] for emotion,f1,recall,precision in zip(dataset.emotions,separate_f1,separate_recall,separate_precision)])
    wandb.log({"Score Table": score_table})
    
    # wandb confustion matrix
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(y_true=all_true_labels, preds=all_pred_labels,class_names=dataset.emotions)})
    
    # Prepare prediction data
    prediction_data = {
        "gender": all_genders,
        "true_emotion": all_true_labels,
        "pred_emotion": all_pred_labels,
    }
    
    if use_dim_emotion:
        reg_loss, avg_ccc_arousal, avg_ccc_valence = dim_criterion(all_pred_av,all_true_av)
        avg_ccc = (avg_ccc_arousal + avg_ccc_valence) / 2
        test_metric.update({
            "test/arousal": avg_ccc_arousal,
            "test/valence": avg_ccc_valence,
            "test/avg_ccc": avg_ccc,
        })
        all_true_av_array = np.array(all_true_av) 
        all_pred_av_array = np.array(all_pred_av) 
        # Add dimensional emotion data
        prediction_data.update({
            "true_EmoAct": all_true_av_array[:,0],
            "true_EmoVal": all_true_av_array[:,1],
            "pred_EmoAct": all_pred_av_array[:,0],
            "pred_EmoVal": all_pred_av_array[:,1],
        })

    wandb.log({**test_metric})
    return prediction_data

def run_test(model_type, device='cuda', **kwargs):
    corpus = kwargs.get('corpus', None)
    project = kwargs.get('project', None)
    ckpt_name = kwargs.get('ckpt_name', None)
    seed = kwargs.get('seed', 42)  
    use_gender = kwargs.get('use_gender', False)  
    batch_size = kwargs.get('batch_size', 16)
    verbose = kwargs.get('verbose', True)

    if ckpt_name == None or corpus == None:
        raise ValueError(f"Checkpoint Name or Corpus Name is Empty.")
    if project == None:
        raise ValueError(f"Project Name is Empty, can't use Weight&Bias")

    wandb.login(key=WANDB_TOKEN)
    run = wandb.init(
        project=project,
        tags=kwargs.get('tags', ["No Tags"]), 
        config = {
            "seed": seed,
            "corpus": corpus,
            "batch_size": batch_size,
            "use_gender": use_gender,
            "dropout": kwargs.get('dropout',0.2),
            "model_type": model_type,
            "upstream_model": kwargs.get('upstream_model',"wav2vec2-base"),
            "finetune_layers": kwargs.get('finetune_layers',1),
            "hidden_dim": kwargs.get('hidden_dim',64),
            "num_layers": kwargs.get('num_layers',2),
            "classifier_output_dim": kwargs.get('classifier_output_dim', 8),
        }
    )

    if model_type == 'UpstreamFinetune':
        model_path = os.path.join(config["PATH_TO_SAVED_MODELS"], ckpt_name)
        model = UpstreamFinetune.from_pretrained(model_path, config["PATH_TO_PRETRAINED_MODELS"], device=device)
    elif model_type == 'UpstreamGender':
        model_path = os.path.join(config["PATH_TO_SAVED_MODELS"], ckpt_name)
        model = UpstreamGender.from_pretrained(model_path, config["PATH_TO_PRETRAINED_MODELS"], device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    print("Load model successfully!")
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Dataset Preparation (Just need to load webdataset)
    use_dim_emotion = True
    if corpus == 'MSPPODCAST':
        dataset = MSPPodcast(config[corpus]['PATH_TO_DATASET']) 
    elif corpus == 'IEMOCAP':
        dataset = IEMOCAP(config[corpus]['PATH_TO_DATASET']) 
    elif corpus == 'CREMAD':
        use_dim_emotion = False
        dataset = CREMAD(config[corpus]['PATH_TO_DATASET']) 
    else:
        raise ValueError(f"Unknown corpus: {corpus}")
    
    test_samples = dataset.test_counts
    test_loader = dataset.create_dataloader('test', batch_size)

    if verbose:
        print(f'There are total {test_samples} test samples')
        numel_list = [p.numel() for p in model.parameters()]
        total_params = sum(numel_list)
        print(f"Total number of model parameters: {total_params:,}")

    length = test_samples // batch_size + 1
    prediction_data = tester(model, test_loader, use_gender, use_dim_emotion, length, dataset)
    
    # Save predictions to CSV
    csv_path = save_predictions_to_csv(dataset, prediction_data, corpus, ckpt_name, use_dim_emotion)
    print(f"Predictions saved to {csv_path}")
    png_path = save_confusion_matrix(prediction_data['pred_emotion'],prediction_data['true_emotion'],dataset.emotions,corpus,ckpt_name)
    print(f"Confusion matrix saved to {png_path}")

    wandb.finish()
    